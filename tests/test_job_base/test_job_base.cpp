#include <include/job_base.h>
#include <gtest/gtest.h>

class Job:public JobBase{
private:

	double value;	
public:
	Job():JobBase(){
		value=0;
	}
	Job(double val):JobBase(){
		value=val;
	}
	virtual __host__ __device__ double getValue() {
		return value;
	}

};
class TestJobBase : public testing::Test
{
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
    
    
public:
	// JobBase jb;
    	Job j;
};

const double *TestJobBase::testSetMsGenePointer(double* ms_gene){
	j.setMsGenePointer(ms_gene);
	return j.ms_gene;
}
const double *TestJobBase::testSetOsSeqGenePointer(double* os_seq_gene){
	j.setOsSeqGenePointer(os_seq_gene);
	return j.os_seq_gene;
}
ProcessTime** TestJobBase::testSetProcessTime(ProcessTime **ptime){
	j.setProcessTime(ptime);
	return j.process_time;
}
double TestJobBase::testSetArrivT(double arriv_time){
	j.setArrivT(arriv_time);
	return j.arriv_t;
}
double TestJobBase::testSetStartTime(double start_time){
	j.setStartTime(start_time);
	return j.start_time;
}

double TestJobBase::testGetMsGene(){
	return j.getMsGene(); 
}
double TestJobBase::testGetOsSeqGene(){
	return j.getOsSeqGene(); 
}
unsigned int TestJobBase::testGetMachineNo(){
	return j.getMachineNo();
}
double TestJobBase::testGetArrivT(){
	return j.getArrivT();
}
double TestJobBase::testGetStartTime(){
	return j.getStartTime();
}
double TestJobBase::testGetEndTime(){
	return j.getEndTime();
	
}
double *x;
double *y;
const double *a;
const double *b;
double qq = 5;
ProcessTime ** z;
TEST_F(TestJobBase, test_JobBase_setMsGenePointer){
    EXPECT_EQ(testSetMsGenePointer(x), a);
    EXPECT_EQ(testSetMsGenePointer(y), b);
}
TEST_F(TestJobBase, test_JobBase_setOsSeqGenePointer){
    EXPECT_EQ(testSetOsSeqGenePointer(x), a);
    EXPECT_EQ(testSetOsSeqGenePointer(y), b);
}
TEST_F(TestJobBase, test_JobBase_setProcessTime){
    EXPECT_EQ(testSetProcessTime(z), z);
    EXPECT_EQ(testSetProcessTime(z), z);
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
    j.setMsGenePointer(&qq);
    EXPECT_EQ(qq, j.getMsGene());
    // EXPECT_EQ(testGetMsGene(), j.getMsGene());
}
TEST_F(TestJobBase, test_JobBase_getOsSeqGene){
    j.setOsSeqGenePointer(&qq);
    EXPECT_EQ(qq, j.getOsSeqGene());
    // EXPECT_EQ(testGetOsSeqGene(), j.getOsSeqGene());
}
TEST_F(TestJobBase, test_JobBase_getMachineNo){
    EXPECT_EQ(testGetMachineNo(), j.getMachineNo());
    EXPECT_EQ(testGetMachineNo(), j.getMachineNo());
}
TEST_F(TestJobBase, test_JobBase_getArrivT){
    EXPECT_EQ(testGetArrivT(), j.getArrivT());
    EXPECT_EQ(testGetArrivT(), j.getArrivT());
}
TEST_F(TestJobBase, test_JobBase_getStartTime){
    EXPECT_EQ(testGetStartTime(), j.getStartTime());
    EXPECT_EQ(testGetStartTime(), j.getStartTime());
}
TEST_F(TestJobBase, test_JobBase_getEndTime){
    EXPECT_EQ(testGetEndTime(), j.getEndTime());
    EXPECT_EQ(testGetEndTime(),j.getEndTime());
}







