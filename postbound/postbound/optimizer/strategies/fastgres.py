
import random
from typing import Optional

from postbound.experiments import workloads
from postbound.qal import qal
from postbound.optimizer import jointree, physops, stages

from postbound.optimizer.strategies.FASTgres_Postbound_Integration import public_api as fga
from postbound.optimizer.strategies.FASTgres_Postbound_Integration import definitions
from postbound.optimizer.strategies.FASTgres_Postbound_Integration.hint_sets import HintSet
from postbound.optimizer.strategies.FASTgres_Postbound_Integration.workloads import workload as wl


def hints_to_operators(hints: HintSet) -> physops.PhysicalOperatorAssignment:
    operators = physops.PhysicalOperatorAssignment()
    # imported order from FASTgres
    fg_list = [physops.JoinOperators.HashJoin,
               physops.JoinOperators.SortMergeJoin,
               physops.JoinOperators.NestedLoopJoin,
               physops.ScanOperators.IndexScan,
               physops.ScanOperators.SequentialScan,
               physops.ScanOperators.IndexOnlyScan]
    operators.global_settings = {fg_list[i]: bool(hints.get(i)) for i in range(len(hints.operators))}
    return operators


def determine_model_file(workload: workloads.Workload) -> str:
    return f"fastgres_model_{workload.name}.model"


class FastgresOperatorSelection(stages.PhysicalOperatorSelection):

    def __init__(self, workload: workloads.Workload, *, rand_seed: float = random.random()) -> None:
        # TODO: fastgres initialization, model loading, sanity checks
        self.workload = workload

        work_load_path = definitions.ROOT_DIR + "/workloads/queries/"
        if "job" in workload.name.lower():
            dbc = definitions.PG_IMDB
            work_load_path += "job/"
        elif "stack" in workload.name.lower():
            dbc = definitions.PG_STACK_OVERFLOW
            work_load_path += "stack/"
        else:
            raise NotImplementedError("Currently only JOB and Stack are supported for FASTgres.")

        fastgres_workload = wl.Workload(work_load_path, workload.name.lower())
        fastgres = fga.Fastgres(fastgres_workload, dbc)
        self.model = fastgres

        random.seed(rand_seed)
        super().__init__()

    def select_physical_operators(self, query: qal.SqlQuery,
                                  join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]
                                  ) -> physops.PhysicalOperatorAssignment:

        query_label = self.workload.label_of(query) + ".sql"
        hints = self.model.predict(query_label)
        assignment = hints_to_operators(hints)
        return assignment

    def describe(self) -> dict:
        return dict()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__

    def close(self):
        self.model.close()
        return
