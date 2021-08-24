#include <gtest/gtest.h>

#define private public
#define protected public

#include "include/machines.h"

#undef protected
#undef private

#include <string>

using namespace std;

class test_machines_t_addNewResource : public testing::Test
{
public:
    machines_t *machines;
    machine_t *m;

    void SetUp() override;
    void TearDown() override;
};

void test_machines_t_addNewResource::SetUp()
{
    machines = new machines_t();
    m = new machine_t();
    m->base.machine_no = stringToInfo("M");
}

void test_machines_t_addNewResource::TearDown()
{
    delete machines;
}

TEST_F(test_machines_t_addNewResource, add_new_resource)
{
    string part_no("PART_NO");
    EXPECT_TRUE(
        machines->_addNewResource(m, part_no, machines->_machines_tools));
    EXPECT_EQ(machines->_machines_tools["M"].size(), 1);
    EXPECT_NE(find(machines->_machines_tools["M"].begin(),
                   machines->_machines_tools["M"].end(), part_no),
              machines->_machines_tools["M"].end());
}

TEST_F(test_machines_t_addNewResource, add_existent_resource)
{
    string part_no("PART_NO");
    EXPECT_TRUE(
        machines->_addNewResource(m, part_no, machines->_machines_tools));
    EXPECT_FALSE(
        machines->_addNewResource(m, part_no, machines->_machines_tools));
    EXPECT_EQ(machines->_machines_tools["M"].size(), 1);
}