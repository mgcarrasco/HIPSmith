#include "HIPSmith/HIPStatement.h"

#include "Block.h"
#include "CGContext.h"
#include "FactMgr.h"
#include "Function.h"
#include "HIPSmith/Globals.h"
#include "HIPSmith/HIPOptions.h"
#include "HIPSmith/StatementHIPSync.h"
#include "OutputMgr.h"
#include "ProbabilityTable.h"
#include "Type.h"
#include "Variable.h"
#include "VariableSelector.h"
#include "VectorFilter.h"
#include "random.h"
#include "util.h"

#include <sstream>
#include <string>
#include <vector>

namespace HIPSmith {
namespace {
DistributionTable *hip_stmt_table = NULL;

const char *print_macro(eSimpleType t) {
  switch (t) {
    case eChar: return "PRINT_INT8";
    case eUChar: return "PRINT_UINT8";
    case eShort: return "PRINT_INT16";
    case eUShort: return "PRINT_UINT16";
    case eInt: return "PRINT_INT";
    case eUInt: return "PRINT_UINT";
    case eLong:
    case eLongLong: return "PRINT_INT64";
    case eULong:
    case eULongLong: return "PRINT_UINT64";
    default: return NULL;
  }
}

void collect_printable(const Variable *v, std::vector<const Variable *> &out) {
  if (v->type && v->type->eType == eStruct) {
    for (const Variable *f : v->field_vars) collect_printable(f, out);
    return;
  }
  if (v->type && print_macro(v->type->simple_type) && !v->isArray &&
      !v->is_volatile() && !v->is_inside_union_field() &&
      !v->is_hip_builtin()) {
    out.push_back(v);
  }
}

unsigned next_print_id = 1;

class StatementPrint : public Statement {
 public:
  StatementPrint(Block *block, const Variable *var, unsigned id)
      : Statement(ePrint, block), var_(var), id_(id) {}

  static StatementPrint *make_random(CGContext &cg_context) {
    if (!HIPOptions::hip_print()) return NULL;
    std::vector<Variable *> visible =
        VariableSelector::find_all_visible_vars(cg_context.get_current_block());
    visible.insert(visible.end(), cg_context.get_current_func()->param.begin(),
                   cg_context.get_current_func()->param.end());
    std::vector<const Variable *> candidates;
    for (const Variable *v : visible) collect_printable(v, candidates);
    const Variable *var = VariableSelector::choose_ok_var(candidates);
    if (!var) return NULL;
    if (!cg_context.check_read_var(var, get_fact_mgr(&cg_context)->global_facts))
      return NULL;
    return new StatementPrint(cg_context.get_current_block(), var,
                              next_print_id++);
  }

  static std::string c_lvalue(const Variable *v) {
    std::ostringstream oss;
    v->Output(oss);
    return oss.str();
  }

  static std::string gdb_how(const Variable *v) {
    const std::string name = c_lvalue(v);
    const size_t arrow = name.find("->");
    if (arrow == std::string::npos) return name;
    const Type *ptr_ty =
        HIPSmith::Globals::GetGlobals()->GetGlobalStructPtrType().ptr_type;
    if (!ptr_ty) return name;
    std::ostringstream ty;
    ptr_ty->Output(ty);
    return "((" + ty.str() + " *)private_lane#(long)" + name.substr(0, arrow) +
           ")->" + name.substr(arrow + 2);
  }

  static std::string c_string_literal(const std::string &s) {
    std::string out = "\"";
    for (char c : s) {
      if (c == '\\' || c == '"') out += '\\';
      out += c;
    }
    out += '"';
    return out;
  }

  void Output(std::ostream &out, FactMgr *, int indent) const override {
    output_tab(out, indent);
    out << print_macro(var_->type->simple_type) << "(";
    var_->Output(out);
    out << ", __LINE__, " << c_string_literal(gdb_how(var_)) << ", " << id_
        << ");" << std::endl;
  }

  bool visit_facts(std::vector<const Fact *> &inputs,
                   CGContext &cg_context) const override {
    bool ok = cg_context.check_read_var(var_, inputs);
    get_fact_mgr(&cg_context)->map_stm_effect[this] =
        cg_context.get_effect_stm();
    return ok;
  }

  void get_blocks(std::vector<const Block *> &) const override {}
  void get_exprs(std::vector<const Expression *> &) const override {}

 private:
  const Variable *var_;
  unsigned id_;
};
}  // namespace

void HIPStatement::InitProbabilityTable() {
  hip_stmt_table = new DistributionTable();
  hip_stmt_table->add_entry(kSync, 10);
  next_print_id = 1;
}

HIPStatement *HIPStatement::make_random(CGContext &cg_context,
                                        enum HIPStatementType st) {
  if (st == kNone) {
    assert(hip_stmt_table != NULL);
    int num = rnd_upto(hip_stmt_table->get_max());
    st = (HIPStatementType)VectorFilter(hip_stmt_table).lookup(num);
  }

  HIPStatement *stmt = NULL;
  switch (st) {
    case kSync:
      if (HIPOptions::hip_sync()) {
        stmt = StatementHIPSync::make_random(cg_context);
      }
      break;
    default:
      assert(false);
  }
  return stmt;
}

Statement *make_random_st(CGContext &cg_context) {
  return HIPStatement::make_random(cg_context, HIPStatement::kNone);
}

Statement *make_random_print(CGContext &cg_context) {
  return StatementPrint::make_random(cg_context);
}

static bool is_print_stmt(const Statement *s) {
  return s && s->get_type() == ePrint;
}

static bool is_same_line_host(const Statement *s) {
  if (!s) return false;
  switch (s->get_type()) {
    case eAssign:
    case eInvoke:
    case eHIPStatement:
      return true;
    default:
      return false;
  }
}

static std::string render_one_line(const Statement *s, FactMgr *fm) {
  std::ostringstream oss;
  s->Output(oss, fm, 0);
  std::string piece = oss.str();
  while (!piece.empty() && (piece.back() == '\n' || piece.back() == '\r'))
    piece.pop_back();
  size_t b = 0;
  while (b < piece.size() && (piece[b] == ' ' || piece[b] == '\t')) ++b;
  piece.erase(0, b);
  return piece;
}

static bool can_glue(const std::vector<const Statement *> &group, FactMgr *fm,
                     std::vector<std::string> *pieces) {
  pieces->clear();
  bool saw_print = false;
  bool saw_host = false;
  for (const Statement *s : group) {
    if (is_print_stmt(s))
      saw_print = true;
    else
      saw_host = true;
    std::string piece = render_one_line(s, fm);
    if (piece.empty() || piece.find('\n') != std::string::npos) return false;
    pieces->push_back(piece);
  }
  return saw_print && saw_host && !pieces->empty();
}

static void emit_statement_normally(const Statement *stm, std::ostream &out,
                                    FactMgr *fm, int indent) {
  stm->pre_output(out, fm, indent);
  stm->Output(out, fm, indent);
  stm->post_output(out, fm, indent);
}

static void emit_glued_group(const std::vector<const Statement *> &group,
                             const std::vector<std::string> &pieces,
                             std::ostream &out, FactMgr *fm, int indent) {
  for (const Statement *s : group) s->pre_output(out, fm, indent);
  output_tab(out, indent);
  for (size_t i = 0; i < pieces.size(); ++i) {
    if (i) out << " ";
    out << pieces[i];
  }
  outputln(out);
  for (const Statement *s : group) s->post_output(out, fm, indent);
}

// Same-line mode never emits a PRINT on its own line. If a skipped print is a
// goto target, keep a labeled empty statement so the jump stays valid.
static void omit_print(const Statement *stm, std::ostream &out, FactMgr *fm,
                       int indent) {
  std::vector<const StatementGoto *> gotos;
  if (!stm->find_jump_sources(gotos)) return;
  stm->pre_output(out, fm, indent);
  output_tab(out, indent);
  out << ";";
  outputln(out);
  stm->post_output(out, fm, indent);
}

void OutputPrintSameLineStatementList(const std::vector<Statement *> &stms,
                                      std::ostream &out, FactMgr *fm,
                                      int indent) {
  size_t i = 0;
  while (i < stms.size()) {
    if (is_print_stmt(stms[i])) {
      std::vector<const Statement *> prints;
      size_t j = i;
      while (j < stms.size() && is_print_stmt(stms[j])) {
        prints.push_back(stms[j]);
        ++j;
      }
      if (j < stms.size() && is_same_line_host(stms[j])) {
        std::vector<const Statement *> group = prints;
        group.push_back(stms[j]);
        std::vector<std::string> pieces;
        if (can_glue(group, fm, &pieces)) {
          emit_glued_group(group, pieces, out, fm, indent);
          i = j + 1;
          continue;
        }
      }
      for (const Statement *p : prints) omit_print(p, out, fm, indent);
      i = j;
      continue;
    }
    emit_statement_normally(stms[i], out, fm, indent);
    ++i;
  }
}

}  // namespace HIPSmith
