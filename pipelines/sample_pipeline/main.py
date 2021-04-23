import numpy as np

from e2ebench import Benchmark, ConfusionMatrixTracker, HyperparameterTracker, BenchmarkSupervisor, TimeMetric, MemoryMetric, PowerMetric, EnergyMetric, ThroughputMetric

bm = Benchmark('sample_db_file.db')

tpm = ThroughputMetric('bloat throughput')
@BenchmarkSupervisor([TimeMetric('bloat time'),
                      MemoryMetric('bloat memory', interval=0.1),
                      PowerMetric('bloat power'),
                      EnergyMetric('bloat energy'),
                      tpm], bm)
def bloat():
    a = []
    for i in range(1, 2):
        a.append(np.random.randn(*([10] * i)))
    print(a)
    tpm.track(420)


def main():
    conf_mat = np.arange(9).reshape((3, 3))
    labels = ['foo', 'bar', 'baz']
    ConfusionMatrixTracker(bm).track(conf_mat, labels, 'foobar')

    with HyperparameterTracker(bm, "hyper params of sample pipeline", ['lr', 'num_epochs', 'num_layers'], 'loss') as ht:
        ht.track({'lr': 0.03, 'num_epochs': 10, 'num_layers': 4, 'loss': 42})
        ht.track({'lr': 0.08, 'num_epochs': 15, 'num_layers': 2, 'loss': 69})

    bloat()

    bm.close()


if __name__ == "__main__":
    main()
