# Statistical summary (scalar fitness)

Comparison uses **`scalar_fitness`**: accuracy minus feature-count penalty (`(k/N)*epsilon`) for NSGA-II best-by-accuracy individual; PSO global-best `fitness`; SGA best individual `accuracy - penalty`.

```
algorithm          case  n_runs  mean_scalar_fitness  std_scalar_fitness  mean_accuracy
    nsga2           zoo       2             0.935417            0.000000       0.979167
      pso           zoo       2             0.936458            0.000000       0.973958
      sga           zoo       2             0.930208            0.008839       0.964583
    nsga2     hepatitis       2             0.908580            0.003683       0.934896
      pso     hepatitis       2             0.911184            0.000000       0.937500
      sga     hepatitis       2             0.905297            0.008325       0.934245
    nsga2 asym-triangle       2             4.983870            0.000000       5.000000
      pso asym-triangle       2             4.983870            0.000000       5.000000
      sga asym-triangle       2             4.951610            0.000000       5.000000
```

