#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#define private public
#define protected public

#include "include/entities.h"
#include "include/entity.h"
#include "include/lot.h"
#include "include/lots.h"
#include "include/machine.h"
#include "include/machines.h"

#undef private
#undef protected

using namespace std;

struct tcs_distribute_resource_t {
    int _number_of_resources;
    map<string, int> _case, _ans;
};

class test_machines_t_distributeAResource
    : public testing::TestWithParam<tcs_distribute_resource_t>
{
public:
    machines_t *machines;

    void SetUp() override;
    void TearDown() override;
};



void test_machines_t_distributeAResource::SetUp()
{
    machines = nullptr;
    machines = new machines_t();
    if (machines == nullptr)
        exit(EXIT_FAILURE);
}

void test_machines_t_distributeAResource::TearDown()
{
    delete machines;
}

TEST_P(test_machines_t_distributeAResource, test_correctness)
{
    auto cs = GetParam();

    map<string, int> ans =
        machines->_distributeAResource(cs._number_of_resources, cs._case);

    EXPECT_EQ(ans.size(), cs._ans.size());
    for (auto it = ans.begin(); it != ans.end(); ++it) {
        EXPECT_NO_THROW(cs._ans.at(it->first));
        EXPECT_EQ(it->second, cs._ans.at(it->first));
    }
}

INSTANTIATE_TEST_SUITE_P(
    test_correctness,
    test_machines_t_distributeAResource,
    testing::Values(
        tcs_distribute_resource_t{
            100,
            {{"g1", 40}, {"g2", 60}},
            {{"g1", 40}, {"g2", 60}}
},

        tcs_distribute_resource_t{50,
                                  {{"g1", 40}, {"g2", 60}},
                                  {{"g1", 20}, {"g2", 30}}},
        tcs_distribute_resource_t{48,
                                  {{"g1", 40}, {"g2", 60}},
                                  {{"g1", 19}, {"g2", 29}}},
        tcs_distribute_resource_t{29,
                                  {{"g1", 43}, {"g2", 31}},

                                  {{"g1", 17}, {"g2", 12}}},

        tcs_distribute_resource_t{3,
                                  {{"g1", 43}, {"g2", 31}},
                                  {{"g1", 2}, {"g2", 1}}},
        tcs_distribute_resource_t{2,
                                  {{"g1", 43}, {"g2", 31}},
                                  {{"g1", 1}, {"g2", 1}}},
        tcs_distribute_resource_t{1,
                                  {
                                      {"g1", 43},
                                      {"g2", 31},
                                  },
                                  {{"g1", 0}, {"g2", 1}}},

        tcs_distribute_resource_t{6,
                                  {{"g1", 1}, {"g2", 12}},
                                  {{"g1", 1}, {"g2", 5}}},
        tcs_distribute_resource_t{5,
                                  {{"g1", 43}, {"g2", 31}, {"g3", 25}},
                                  {{"g1", 3}, {"g2", 1}, {"g3", 1}}},
        tcs_distribute_resource_t{3,
                                  {{"g1", 43}, {"g2", 1}, {"g3", 5}},
                                  {{"g1", 1}, {"g2", 1}, {"g3", 1}}}));
