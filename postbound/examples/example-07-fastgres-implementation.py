#!/usr/bin/env python3
#
# This script demonstrates how a learned optimizer can be used with PostBOUND. More specifically, it demonstrates how to
# use FASTgres for physical operator prediction utilizing hint sets.
#
# Requirements: A running IMDB instance on Postgres with the connect file being set-up correctly. This can be achieved using
# the utilities in the root postgres directory. Additionally, a trained FASTgres model for the JOB workload.
#

import math

from postbound import postbound as pb
from postbound.db import postgres
from postbound.experiments import workloads, runner
from postbound.optimizer.strategies import fastgres

# Setup: we optimize queries from the Join Order Benchmark on a Postgres database
postgres_db = postgres.connect(connect_string="dbname=imdb user=postgres password=postgres host=localhost port=5432")
job_workload = workloads.job()

# Obtain a training and test split
n_train_queries = math.floor(0.2 * len(job_workload))
# only test_queries will be needed
train_queries = job_workload.pick_random(n_train_queries)
test_queries = job_workload - train_queries

fastgres_recommender = fastgres.FastgresOperatorSelection(job_workload)

pipeline = pb.TwoStageOptimizationPipeline(postgres_db)
pipeline.setup_physical_operator_selection(fastgres_recommender)
pipeline.build()

result_df = runner.optimize_and_execute_workload(test_queries, pipeline)

print("Benchmark results:")
print(result_df)
