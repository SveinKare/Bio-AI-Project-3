# Statistical summary (scalar fitness)

Comparison uses **`scalar_fitness`**: accuracy minus feature-count penalty (`(k/N)*epsilon`) for NSGA-II best-by-accuracy individual; PSO global-best `fitness`; SGA best individual `accuracy - penalty`.

```
algorithm     case  n_runs  mean_scalar_fitness  std_scalar_fitness  mean_accuracy
    nsga2   breast     100             0.892344            0.000000       0.970122
      pso   breast     100             0.933418            0.000000       0.955640
      sga   breast     100             0.933387            0.000113       0.956387
    nsga2   credit     100             0.830653            0.003484       0.891253
      pso   credit     100             0.857840            0.000000       0.871173
      sga   credit     100             0.857556            0.001040       0.872289
    nsga2   letter     100             0.887583            0.000000       0.956333
      pso   letter     100             0.891094            0.000000       0.947344
      sga   letter     100             0.891094            0.000000       0.947344
    nsga2 triangle     100             3.975000            0.000000       4.000000
      pso triangle     100             3.975000            0.000000       4.000000
      sga triangle     100             3.975000            0.000000       4.000000
```

