#pragma once
#include <vkbase_core.hpp>


enum class ImageState
{
    ReadyForDraw,
    Draw,
    ReadyToPresent,
    Present
};


class CudaVulkanSwapChain
{
    uint32_t m_number_of_images;
    uint32_t m_width=1920;
    uint32_t m_height=1080;

    struct ImageData
    {
        vkbase::cuda::Image image;
        std::unique_ptr<std::atomic<ImageState>> state;
        vkbase::imgui::Image image_imgui;

        vkbase::CommandBufferGraphics transfer_General_to_Optimal;
        vkbase::CommandBufferGraphics transfer_Optimal_to_General;
    };







public:
    std::vector<std::unique_ptr<ImageData>> images;

    int currently_presented=-1;
    int last_drawn_image=0;
    bool  present_flag=false;

    ImageData createImage(uint32_t width, uint32_t height)
    {
        vkbase::cuda::Image image{m_width,m_height};
            vkbase::imgui::Image imgui_image(image.image_view);
            auto state=std::make_unique<std::atomic<ImageState>>(ImageState::ReadyForDraw);

            VkImageMemoryBarrier barrier = {};

            vkbase::CommandBufferGraphics cb_general_to_optimal;
            cb_general_to_optimal.begin();
            {
                // --- BARRIER 1: Prepare the New Texture for Vulkan to Read ---
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL; // CUDA left it in GENERAL
                barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.srcAccessMask = 0; // CUDA already finished via host sync
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barrier.image = image.image;
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.layerCount = 1;


                vkCmdPipelineBarrier(
                    cb_general_to_optimal,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &barrier // Pass the array of two barriers
                );
            }
            cb_general_to_optimal.end();


            vkbase::CommandBufferGraphics cb_optimal_to_general;
            cb_optimal_to_general.begin();
            {
                // --- BARRIER 2: Prepare the Old Texture for CUDA to Write ---
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // Vulkan was reading it
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT; // Wait for Vulkan to finish reading
                barrier.dstAccessMask = 0; // External API (CUDA) will handle the writing
                barrier.image = image.image;
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.layerCount = 1;

                vkCmdPipelineBarrier(
                    cb_optimal_to_general,
                    // Source stage: When Vulkan finishes reading the old texture in the fragment shader
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    // Destination stage: The very next fragment shader execution for the new texture
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &barrier // Pass the array of two barriers
                );
            }
            cb_optimal_to_general.end();

        return ImageData{
            std::move(image),std::move(state),std::move(imgui_image),std::move(cb_general_to_optimal),
            std::move(cb_optimal_to_general)
        };
    }


    CudaVulkanSwapChain(uint32_t number_of_images)
    {
        m_number_of_images=number_of_images;

        images.clear();
        last_drawn_image=0;
        for (int i=0;i<m_number_of_images;++i)
        {
            images.push_back(std::make_unique<ImageData>(createImage(m_width,m_height)));
        }
    }


    std::mutex image_mutex;
    uint32_t acquireNextImage()
    {
        auto id= 0;
        while (true)
        {
            std::lock_guard lock(image_mutex);
            auto image=images[id].get();
            if (image!=nullptr && image->state->load()==ImageState::ReadyForDraw)
            {
                *images[id]->state=ImageState::Draw;
                vkbase::info(std::format("      {} Acquired image {} ",vkbase::sys::getMilliseconds(),id));
                return id;
            }



            id=(id+1)%images.size();
        };



    }

    vkbase::cuda::Image& getImage(uint32_t id)
    {
        std::lock_guard lock(image_mutex);
        return images[id]->image;
    }


    void present(uint32_t image_id)
    {
        std::lock_guard lock(image_mutex);
        vkbase::info(std::format("      {} Mark image {} for present ",vkbase::sys::getMilliseconds(),image_id));
        *images[image_id]->state=ImageState::ReadyToPresent;
        last_drawn_image = image_id;
        present_flag=true;
    }

    void printImageStates()
    {
        std::cout<<vkbase::sys::getMilliseconds()<<"   ";
        for (int i=0;i<images.size();++i)
        {
            std::cout<<i<<") ["<<(int)images[i]->state->load()<<"]    ";
        }
        std::cout<<"\n";
    }



    void setImageSize(uint32_t width, uint32_t height)
    {
        std::lock_guard lock(image_mutex);
        if (width<1||width>10000)
            width=1;
        if (height<1||height>10000)
            height=1;
        m_width=width;
        m_height=height;
    }

    std::vector<vkbase::WaitPoint> swap()
    {
        std::lock_guard lock(image_mutex);
        printImageStates();

        if (present_flag)
        {
            present_flag=false;
            auto image_id=last_drawn_image;

            auto newImage= images[image_id].get();
            for (auto& img:images)
            {
                if (img->state->load()==ImageState::ReadyToPresent)
                {
                    if (img.get()==newImage)//set new image as being presented
                    {
                        *img->state=ImageState::Present;
                    }
                    else//in case more than one image was rendered by this time set them free
                    {
                        *img->state=ImageState::ReadyForDraw;
                    }
                }
            }

            vkbase::info(std::format("{} Swapped old: {} new: {}",vkbase::sys::getMilliseconds(),currently_presented,image_id));


            auto newImageReady=newImage->transfer_General_to_Optimal.submit_async();

            //exclude first swap when there is no old image
            if (currently_presented>=0)
            {
                auto& img=*images[currently_presented];
                if (img.image.width!=m_width||img.image.height!=m_height)
                {
                    auto old_image=std::move(images[currently_presented]);
                    images[currently_presented] = std::make_unique<ImageData>(createImage(m_width,m_height));

                    vkbase::info(std::format("{} Recreated {}",vkbase::sys::getMilliseconds(),currently_presented));

                    auto lastFrameToken=vkbase::getPreviousFrameWaitToken();
                    vkbase::doAfter(lastFrameToken,[old_image=std::move(old_image),id=currently_presented]()mutable
                    {
                        vkbase::info(std::format("{} Destroyed: {}",vkbase::sys::getMilliseconds(),id));
                    });
                }
                else
                {
                    auto image_is_free= img.transfer_Optimal_to_General.submit_async(
                        {vkbase::WaitPoint{vkbase::getPreviousFrameWaitToken(),VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}});


                    std::thread returnImageStatus([image_is_free,&img,id=currently_presented]()
                    {
                        vkbase::info(std::format("{} Scheduled for Release: {} frame value={}",vkbase::sys::getMilliseconds(),id,image_is_free.value));
                        image_is_free.wait();

                        *img.state=ImageState::ReadyForDraw;
                        vkbase::info(std::format("{} Released: {}",vkbase::sys::getMilliseconds(),id));
                    });
                    returnImageStatus.detach();
                }
            }

            currently_presented=image_id;
            vkbase::info(std::format("{} : currently_presented={}",vkbase::sys::getMilliseconds(),currently_presented));
            return {vkbase::WaitPoint{newImageReady,VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT}};
        }
        printImageStates();

        return {};
    }

    ImageData* getImageToPresent()
    {
        std::lock_guard lock(image_mutex);
        if (currently_presented==-1)
            return  nullptr;
        vkbase::info(std::format("{} : Requested {}",vkbase::sys::getMilliseconds(),currently_presented));
        return images[currently_presented].get();
    }
};


