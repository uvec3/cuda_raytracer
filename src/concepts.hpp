#pragma once

template <typename R, typename T>
concept range_of = std::ranges::range<R> && std::same_as<std::ranges::range_value_t<R>, T>;