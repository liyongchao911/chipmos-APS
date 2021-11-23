#ifndef __MACHINE_CONSTRAINT_R__
#define __MACHINE_CONSTRAINT_R__

#include "include/machine_constraint.h"

class machine_constraint_r_t : public machine_constraint_t
{
private:
    machine_constraint_r_t(){};

protected:
    bool _isMachineRestrained(std::regex entity_re,
                              std::string restrained_model,
                              std::string entity_name,
                              std::string machine_model) override;

public:
    explicit machine_constraint_r_t(csv_t csv);
};

#endif
