
import random
from typing import Optional

from postbound.experiments import workloads
from postbound.qal import qal
from postbound.optimizer import jointree, physops, stages

from postbound.optimizer.strategies.FASTgres_Postbound_Integration import public_api as fga
from postbound.optimizer.strategies.FASTgres_Postbound_Integration import definitions
from postbound.optimizer.strategies.FASTgres_Postbound_Integration.hint_sets import HintSet


def hints_to_operators(hints: int) -> physops.PhysicalOperatorAssignment:
    hint_set = HintSet(hints)
    operators = physops.PhysicalOperatorAssignment()
    operators.global_settings = {hint_set.get_name(i): bool(hint_set.get(i)) for i in range(len(hint_set.operators))}
    return operators


def determine_model_file(workload: workloads.Workload) -> str:
    return f"fastgres_model_{workload.name}.model"


def load_model_from_path(path: str):
    pass


class FastgresOperatorSelection(stages.PhysicalOperatorSelection):

    def __init__(self, workload: workloads.Workload, *, rand_seed: float = random.random()) -> None:
        # TODO: fastgres initialization, model loading, sanity checks

        if "job" in workload.name:
            dbc = definitions.PG_IMDB
        elif "stack" in workload.name:
            dbc = definitions.PG_STACK_OVERFLOW
        else:
            raise NotImplementedError("Currently only JOB and Stack are supported for FASTgres.")

        fastgres = fga.Fastgres(workload, dbc)
        self.model = fastgres

        random.seed(rand_seed)
        super().__init__()

    def select_physical_operators(self, query: qal.SqlQuery,
                                  join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]
                                  ) -> physops.PhysicalOperatorAssignment:
        hints = self.model.predict("1a.sql")
        assignment = hints_to_operators(hints)
        return assignment
