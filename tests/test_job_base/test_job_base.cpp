#include <include/job_base.h>
#include <gtest/gtest.h>


class TestJobBase : public testing::Test
{
protected:
    int testSetNumber(int);
private:
    JobBase jb;
};

int TestJobBase::testSetNumber(int number){
    jb.setNumber(number);
    return jb.number;
}

TEST_F(TestJobBase, test_JobBase_setNumber){
    EXPECT_EQ(testSetNumber(5), 5);
    EXPECT_EQ(testSetNumber(10), 10);
}