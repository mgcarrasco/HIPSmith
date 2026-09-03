#ifndef _HIPSMITH_HIPSTATEMENT_H_
#define _HIPSMITH_HIPSTATEMENT_H_

#include <iosfwd>
#include <ostream>
#include <vector>

#include "CommonMacros.h"
#include "Statement.h"

class CGContext;
class FactMgr;

namespace HIPSmith {

class HIPStatement : public Statement {
 public:
  enum HIPStatementType {
    kNone = 0,  // Sentinel value
    kSync,
  };

  HIPStatement(HIPStatementType type, Block* block)
      : Statement(eHIPStatement, block), hip_statement_type_(type) {}
  HIPStatement(HIPStatement&& other) = default;
  HIPStatement& operator=(HIPStatement&& other) = default;
  virtual ~HIPStatement() {}

  // Factory for creating a random HIP statement.
  static HIPStatement* make_random(CGContext& cg_context,
                                   enum HIPStatementType st);

  // Initialise the probability table for selecting a random statement.
  static void InitProbabilityTable();

  enum HIPStatementType GetHIPStatementType() const {
    return hip_statement_type_;
  }

 private:
  HIPStatementType hip_statement_type_;

  DISALLOW_COPY_AND_ASSIGN(HIPStatement);
};

// Hook method called by Csmith's Statement::make_random
Statement* make_random_st(CGContext& cg_context);
Statement* make_random_print(CGContext& cg_context);

// When --hip-print-same-line is set, emit PRINT_* on the same source line as a
// neighboring assign, call, or HIP fence. Omit a PRINT that cannot share a
// line with such a host (including a print next to return/if/for/block).
void OutputPrintSameLineStatementList(const std::vector<Statement*>& stms,
                                      std::ostream& out, FactMgr* fm,
                                      int indent);

}  // namespace HIPSmith

#endif  // _HIPSMITH_HIPSTATEMENT_H_