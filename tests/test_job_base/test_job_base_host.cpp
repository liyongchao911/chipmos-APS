#include <include/job_base.h>
#include <gtest/gtest.h>

#include "test_job_base.h"


class TestJobBase : public testing::Test {
protected:
	const double *testSetMsGenePointer(double * ms_gene);
	const double *testSetOsSeqGenePointer(double * os_seq_gene);
	ProcessTime** testSetProcessTime(ProcessTime **process_timetime);
	double testSetArrivT(double arriv_time);
	double testSetStartTime(double start_time);
	double testGetMsGene();
	double testGetOsSeqGene();
	unsigned int testGetMachineNo();
	double testGetArrivT();
	double testGetStartTime();
	double testGetEndTime();
	
	void SetUp() override;
    
public:
	// JobBase jb;
    	Job *j;
};

void TestJobBase::SetUp(){
	j = newJob(100);
}

const double *TestJobBase::testSetMsGenePointer(double* ms_gene){
	// j->setMsGenePointer(ms_gene);
	j->base.setMsGenePointer(&j->base, ms_gene);
	return j->base.ms_gene;
}
const double *TestJobBase::testSetOsSeqGenePointer(double* os_seq_gene){
	// j.setOsSeqGenePointer(os_seq_gene);
	j->base.setOsSeqGenePointer(&j->base, os_seq_gene);
	return j->base.os_seq_gene;
}
ProcessTime** TestJobBase::testSetProcessTime(ProcessTime **ptime){
	// j.setProcessTime(ptime);
	j->base.setProcessTime(&j->base, ptime, 0);
	return j->base.process_time;
}
double TestJobBase::testSetArrivT(double arriv_time){
	j->base.setArrivT(&j->base, arriv_time);
	return j->base.arriv_t;
}
double TestJobBase::testSetStartTime(double start_time){
	j->base.setStartTime(&j->base, start_time);
	return j->base.start_time;
}

double TestJobBase::testGetMsGene(){
	return j->base.getMsGene(&j->base); 
}
double TestJobBase::testGetOsSeqGene(){
	return j->base.getOsSeqGene(&j->base); 
}
unsigned int TestJobBase::testGetMachineNo(){
	return j->base.getMachineNo(&j->base);
}
double TestJobBase::testGetArrivT(){
	return j->base.getArrivT(&j->base);
}
double TestJobBase::testGetStartTime(){
	return j->base.getStartTime(&j->base);
}
double TestJobBase::testGetEndTime(){
	return j->base.getEndTime(&j->base);
	
}
double *x;
double *y;
const double *a;
const double *b;
double qq = 5;
ProcessTime ** z;
TEST_F(TestJobBase, test_JobBase_setMsGenePointer){
    EXPECT_EQ(testSetMsGenePointer(x), x);
    EXPECT_EQ(testSetMsGenePointer(y), y);
}
TEST_F(TestJobBase, test_JobBase_setOsSeqGenePointer){
    EXPECT_EQ(testSetOsSeqGenePointer(x), x);
    EXPECT_EQ(testSetOsSeqGenePointer(y), y);
}
TEST_F(TestJobBase, test_JobBase_setProcessTime){
    EXPECT_EQ(testSetProcessTime(z), z);
    // EXPECT_EQ(testSetProcessTime(z), z);
}
TEST_F(TestJobBase, test_JobBase_setArrivT){
    EXPECT_EQ(testSetArrivT(5), 5);
    EXPECT_EQ(testSetArrivT(10), 10);
}
TEST_F(TestJobBase, test_JobBase_setStartTime){
    EXPECT_EQ(testSetStartTime(5), 5);
    EXPECT_EQ(testSetStartTime(10), 10);
}
TEST_F(TestJobBase, test_JobBase_getMsGene){
    j->base.setMsGenePointer(&j->base, &qq);
    EXPECT_EQ(qq, j->base.getMsGene(&j->base));
    // EXPECT_EQ(testGetMsGene(), j.getMsGene());
}
TEST_F(TestJobBase, test_JobBase_getOsSeqGene){
    j->base.setOsSeqGenePointer(&j->base, &qq);
    EXPECT_EQ(qq, j->base.getOsSeqGene(&j->base));
    // EXPECT_EQ(testGetOsSeqGene(), j.getOsSeqGene());
}
TEST_F(TestJobBase, test_JobBase_getMachineNo){
    EXPECT_EQ(testGetMachineNo(), j->base.getMachineNo(&j->base));
    // EXPECT_EQ(testGetMachineNo(), j.getMachineNo());
}
TEST_F(TestJobBase, test_JobBase_getArrivT){
    EXPECT_EQ(testGetArrivT(), j->base.getArrivT(&j->base));
    // EXPECT_EQ(testGetArrivT(), j.getArrivT());
}
TEST_F(TestJobBase, test_JobBase_getStartTime){
    EXPECT_EQ(testGetStartTime(), j->base.getStartTime(&j->base));
    // EXPECT_EQ(testGetStartTime(), j.getStartTime());
}
TEST_F(TestJobBase, test_JobBase_getEndTime){
    EXPECT_EQ(testGetEndTime(), j->base.getEndTime(&j->base));
    // EXPECT_EQ(testGetEndTime(),j.getEndTime());
}

