#!/bin/bash

spark-submit --class br.ufrn.dimap.forall.spark.SparkOverflow --conf "spark.cores.max=24" --conf "spark.executor.cores=4" --conf "spark.executor.memory=3g" --conf "spark.driver.memory=6g" --conf "spark.driver.cores=8" target/scala-2.11/SparkOverflow-assembly-1.0.jar > logs/output.log.`date +%y%m%d_%H%M` 2> logs/error.log
