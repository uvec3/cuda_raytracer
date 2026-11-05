#pragma once
#include <concepts>

class ContainerExplorer;

template <typename T>
concept ASDeviceView = requires (T as_view)//Acceleration structure device view concept for static type checking
{
    { as_view.traverse() } -> std::same_as<int>;
};


template <typename T>
concept AS = requires (T as, ContainerExplorer& explorer)//Acceleration structure concept for static type checking
{
    { as.device_view() } -> ASDeviceView;
    as.register_containers(explorer);
};
