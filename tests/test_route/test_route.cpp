#include "tests/test_route/test_route.h"

namespace test_route
{
route_t * ::test_route::test_route_base_t::route = nullptr;
csv_t * ::test_route::test_route_base_t::__routelist = nullptr;
csv_t * ::test_route::test_route_base_t::__queeu_time = nullptr;
csv_t * ::test_route::test_route_base_t::__process_find_lot_size_and_entity =
    nullptr;
csv_t * ::test_route::test_route_base_t::__cure_time = nullptr;
}  // namespace test_route
