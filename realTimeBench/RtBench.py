from tensorflow.keras import callbacks
import time
import psutil
import pyRAPL
import os
from math import ceil
import sqlalchemy as db
from datetime import datetime
from realTimeBench.webapp import WebApp
from multiprocessing import Process, Manager
from pytorch_lightning.callbacks import Callback


def insert_live_metrics(metrics, loss, loss_trend, acc, acc_trend, memory, current_batch,
                      time, energy, cpu, phase, current_epoch=0, no_epochs=0, eta_epoch=0, eta_train=0, no_batches=0):
    """
        Updates the live metrics.

        Parameters
        ----------
        metrics : dict
            Contains the live metrics.
        loss : float
            The current loss.
        loss_trend : int
            The current loss trend.
        acc : float
            The current accuracy.
        acc_trend : int
            The current accuracy trend.
        memory : float
            The current memory usage.
        current_batch : int
            The current batch
        time : float
            The time spent so far.
        energy : float
            The current energy usage.
        cpu : float
            The current CPU usage.
        phase : String
            Training or Testing phase.
        current_epoch : int
            The current epoch number.
        no_epochs : int
            The total amount of epochs.
        eta_epoch : float
            The ETA to finish the epoch.
        eta_train : float
            The ETA to finish the trainig phase.
        no_batches : int
            The total number of batches.

        Returns
        -------
        metrics : dict
            Contains the live metrics
    """
    metrics["phase"] = phase
    metrics["epoch"] = current_epoch
    metrics["no_epochs"] = no_epochs
    metrics["batch"] = current_batch
    metrics["no_batches"] = no_batches
    metrics["loss"] = loss
    metrics["loss_trend"] = loss_trend
    metrics["acc"] = acc
    metrics["acc_trend"] = acc_trend
    metrics["memory"] = memory
    metrics["time"] = time
    metrics["eta_epoch"] = eta_epoch
    metrics["eta_train"] = eta_train
    metrics["energy"] = energy
    metrics["cpu"] = cpu
    return metrics


def insert_epoch_summary(metrics, current_epoch, time, avg_time, loss, loss_trend, acc, acc_trend):
    """
        Updates the epoch summaries' metrics.

        Parameters
        ----------
        metrics : dict
            Contains the epoch summaries' metrics.
        current_epoch : int
            The current epoch number.
        time : float
            The time spent in the epoch.
        avg_time : float
            The average time spent on a batch.
        loss : float
            The loss of the epoch.
        loss_trend : int
            The loss trend of the epoch.
        acc : float
            The accuracy of the epoch.
        acc_trend : int
            The accuracy trend of the epoch.

        Returns
        -------
        metrics : dict
            Contains the epoch summaries' metrics
    """
    metrics["epoch"] = current_epoch
    metrics["time"] = time
    metrics["avg_time"] = avg_time
    metrics["loss"] = loss
    metrics["loss_trend"] = loss_trend
    metrics["acc"] = acc
    metrics["acc_trend"] = acc_trend
    return metrics


def insert_test_summary(metrics, loss, acc, loss_trend, acc_trend):
    """
        Updates the test summary.

        Parameters
        ----------
        metrics : dict
            Contains the test summaries' metrics.
        loss : float
            The test loss.
        acc : float
            The test accuracy.
        loss_trend : int
            The test loss trend.
        acc_trend : int
            The test accuracy trend.

        Returns
        -------
        metrics : dict
            Contains the test summaries' metrics.
    """
    metrics["test_loss"] = loss
    metrics["test_acc"] = acc
    metrics["test_loss_trend"] = loss_trend
    metrics["test_acc_trend"] = acc_trend
    return metrics


def insert_comparison(metrics, ResultSet):
    """
        Updates the comparison run's metrics.

        Parameters
        ----------
        metrics : dict
            Contains the comparisons run's metrics.
        ResultSet : ResultSet
            Contains the log table's entries for the comparison run.

        Returns
        -------
        metrics : dict
            Contains the comparisons run's metrics.
    """
    if ResultSet is not None:
        metrics["available"] = True
        metrics["time"] = ResultSet[1]
        metrics["epoch"] = ResultSet[2]
        metrics["batch"] = ResultSet[3]
        metrics["loss"] = ResultSet[4]
        metrics["loss_trend"] = ResultSet[5]
        metrics["acc"] = ResultSet[6]
        metrics["acc_trend"] = ResultSet[7]
        metrics["memory"] = ResultSet[8]
        metrics["cpu"] = ResultSet[9]
        metrics["energy"] = ResultSet[10]
    else:
        metrics["available"] = False
    return metrics


def setup_webapp(metrics):
    """
        Initializes the webapp.

        Parameters
        ----------
        metrics : dict
            Contains the shared metrics.

        Returns
        -------
        webapp : WebApp
            The corresponding webapp for visualization.
    """
    webApp = WebApp()
    server_process = Process(target=webApp.run, args=(metrics,))
    server_process.start()
    time.sleep(0.5)
    return webApp


def setup_table(metadata):
    """
        Initializes the logs table.

        Parameters
        ----------
        metadata : db.Metadata
            Contains the table's metadata.

        Returns
        -------
        table : db.Table
            The logs table.
    """
    return db.Table('logs', metadata,
                       db.Column('Starttime', db.String()),
                       db.Column('Time', db.Float()),
                       db.Column('Epoch', db.Integer()),
                       db.Column('Batch', db.Integer()),
                       db.Column('Loss', db.Float()),
                       db.Column('LossTrend', db.Float()),
                       db.Column('Accuracy', db.Float()),
                       db.Column('AccuracyTrend', db.Float()),
                       db.Column('Memory', db.Float()),
                       db.Column('CPU', db.Float()),
                       db.Column('Power', db.Float()))


class RtBenchPytorch(Callback):
    """
        The PyTorch version of the main Benchmarking tool.

        Attributes
        ----------
        connection : db.Connection
            The connection to the database engine.
        logs : db.Table
            The database table for logging the metrics.
        run_begin_time : datetime
            The starttime of the run. Used to uniquely identify a run.
        first_loss : int
            The first loss of an epoch. Needed for trend comparison.
        first_acc : int
            The first accuracy of an epoch. Needed for trend comparison.
        train_begin_time : float
            The starttime of the training phase.
        epoch_begin_time : float
            The starttime of the epoch.
        test_begin_time : float
            The starttime of the test phase.
        interval : float
            Interval tracker for consistent memory and energy measuring.
        energy : float
            Amount of energy used.
        memory : float
            Amount of memory used.
        webapp : WebApp
            The corresponding webapp for visualization.
        comp_train_loss : float
            The training loss of the previous epoch. Needed for trend in the epoch summaries.
        comp_train_acc : float
            The training accuracy of the previous epoch. Needed for trend in the epoch summaries.
        comp_test_loss : float
            The testing loss of the previous epoch. Needed for trend in the epoch summaries.
        comp_test_acc : float
            The testing accuracy of the previous epoch. Needed for trend in the epoch summaries.
        no_batches : int
            The total number of batches.
        no_epochs : int
            The total number of epochs.
        metrics : dict
            All the key metrics to be shared between the webapp and RtBench.
        process : Process
            The psutil process for CPU and memory measurement.
        meter : Measurement
            The pyRAPL meter for energy measurement.
    """
    def __init__(self):
        engine = db.create_engine('sqlite:///logs.db?check_same_thread=False')
        self.connection = engine.connect()
        metadata = db.MetaData()
        self.logs = setup_table(metadata)
        metadata.create_all(engine)

        self.run_begin_time = self.first_loss = self.first_acc = \
            self.train_begin_time = self.epoch_begin_time = self.test_begin_time = self.interval = \
            self.energy = self.memory = self.webApp = self.comp_train_loss = self.comp_train_acc = \
            self.comp_test_loss = self.comp_test_acc = self.no_batches = self.no_epochs = 0

        manager = Manager()
        self.metrics = manager.dict()
        live_keys = ["epoch", "no_epochs", "batch", "no_batches", "loss", "loss_trend", "acc", "acc_trend",
                     "memory", "time", "eta_epoch", "eta_train", "energy", "cpu", "phase"]
        epoch_keys = ["epoch", "loss", "loss_trend", "acc", "acc_trend", "time", "avg_time", "test_loss", "test_acc",
                      "test_loss_trend", "test_acc_trend"]
        comparison_keys = ["time", "epoch", "batch", "loss", "loss_trend", "acc", "acc_trend",
                           "memory", "cpu", "energy", "available"]
        self.metrics["live"] = dict.fromkeys(live_keys)
        self.metrics["epoch"] = dict.fromkeys(epoch_keys)
        self.metrics["batch_update"] = 1
        self.metrics["no_batches"] = self.no_batches
        self.metrics["comparison_choice"] = ""
        self.metrics["comparison_options"] = self.get_comparison_options()
        self.metrics["comparison"] = dict.fromkeys(comparison_keys)
        self.metrics["comparison"]["available"] = False

        self.process = psutil.Process(os.getpid())
        pyRAPL.setup()
        self.meter = pyRAPL.Measurement('bar')

    def get_comparison_options(self):
        """
            Returns all the available runs available for comparison.

            Returns
            -------
            results : datetime[]
                Array of all the available comparison runs in the database.
        """
        query = db.select([self.logs.columns.Starttime.distinct()])
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.fetchall()
        return [r[0] for r in ResultSet]

    def get_comparison(self):
        """
            Updates the comparison run depending on the selection and the current progress of the live run.
        """
        time_comp = time.perf_counter() - self.train_begin_time
        query = db.select([self.logs]).where(db.and_(self.logs.columns.Starttime ==
                                                     self.metrics["comparison_choice"],
                                                     self.logs.columns.Time > time_comp)).order_by(
            db.asc(self.logs.columns.Time))
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.first()

        self.metrics["comparison"] = insert_comparison(self.metrics["comparison"], ResultSet)

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
            The callback for the start of the training.

            Parameters
            ----------
            trainer : pl.Trainer
                The pyTorch trainer. Needed to get the amount of epochs and batches.
        """
        self.no_epochs = self.metrics["no_epochs"] = trainer.max_epochs
        self.no_batches = self.metrics["no_batches"] = trainer.num_training_batches + trainer.num_val_batches[0]
        self.webApp = setup_webapp(metrics=self.metrics)
        self.train_begin_time = time.perf_counter()
        self.run_begin_time = datetime.now().replace(microsecond=0).isoformat(' ')
        self.interval = time.time()
        self.meter.begin()

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
            The callback for the start of an epoch.
        """
        self.epoch_begin_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
            The callback for the end of a batch.

            Parameters
            ----------
            trainer : pl.Trainer
                The pyTorch trainer. Needed to get current epoch.
            outputs : dict
                Contains the metrics.
            batch_idx : int
                The current batch number.
        """
        if batch_idx%self.metrics["batch_update"] != 0:
            return
        if time.time() - self.interval > 1:
            self.memory = self.process.memory_info()[0] / (2 ** 20)
            self.meter.end()
            self.energy = self.meter.result.pkg[0] * 0.00000000028
            self.interval = time.time()
            self.meter.begin()

        self.get_comparison()
        loss = outputs["loss"].item()
        acc = outputs["acc"].item()
        time_epoch = time.perf_counter() - self.epoch_begin_time
        time_train = time.perf_counter() - self.train_begin_time
        eta_epoch = (time_epoch * self.no_batches / (batch_idx+1)) - time_epoch
        eta_train = (time_train * (self.no_epochs * self.no_batches /
                           (batch_idx + 1 + trainer.current_epoch * self.no_batches))) - time_train
        if batch_idx == 0 and trainer.current_epoch == 0:
            self.first_loss = loss
            self.first_acc = acc
        if self.first_loss - loss != 0:
            loss_trend = int((loss - self.first_loss)/self.first_loss * 100)
        else:
            loss_trend = 0

        if self.first_acc - acc != 0:
            acc_trend = int((acc - self.first_acc)/self.first_acc * 100)
        else:
            acc_trend = 0

        query = db.insert(self.logs).values(Starttime=self.run_begin_time, Time="%.1f" % time_train, Epoch=trainer.current_epoch + 1,
                                            Batch=batch_idx + 1, Loss="%.3f" % loss, LossTrend=loss_trend,
                                            Accuracy="%.3f" % acc, AccuracyTrend=acc_trend, Memory="%.1f" % self.memory,
                                            CPU=psutil.cpu_percent(),
                                            Power="%.4f" % (self.energy))
        self.connection.execute(query)
        self.connection.execute(db.select([self.logs])).fetchall()

        self.metrics["live"] = insert_live_metrics(metrics=self.metrics["live"],
                                                   phase="train",
                                                   current_epoch=trainer.current_epoch + 1,
                                                   no_epochs=self.no_epochs,
                                                   current_batch=batch_idx + 1,
                                                   no_batches=self.no_batches,
                                                   loss=loss,
                                                   loss_trend=loss_trend,
                                                   acc=acc,
                                                   acc_trend=acc_trend,
                                                   memory=self.memory,
                                                   time=time_train,
                                                   eta_epoch=eta_epoch,
                                                   eta_train=eta_train,
                                                   energy=self.energy,
                                                   cpu=psutil.cpu_percent())

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        """
            The callback for the end of an epoch.

            Parameters
            ----------
            trainer : pl.Trainer
                The pyTorch trainer. Needed to get the loss and accuracy and current epoch.
        """
        loss = trainer.logged_metrics['loss_epoch'].item()
        acc = trainer.logged_metrics['acc_epoch'].item()
        time_epoch = time.perf_counter() - self.epoch_begin_time
        loss_trend = acc_trend = 0

        if trainer.current_epoch != 0:
            if self.comp_train_loss - loss != 0 and self.comp_train_loss != 0:
                loss_trend = int((loss - self.comp_train_loss) / self.comp_train_loss * 100)
            if self.comp_train_acc - acc != 0 and self.comp_train_acc != 0:
                acc_trend = int((acc - self.comp_train_acc) / self.comp_train_acc * 100)

        self.comp_train_loss = loss
        self.comp_train_acc = acc

        self.metrics["epoch"] = insert_epoch_summary(metrics=self.metrics["epoch"],
                                                     current_epoch=trainer.current_epoch + 1,
                                                     time=time_epoch,
                                                     loss=loss,
                                                     avg_time=time_epoch / self.no_batches,
                                                     loss_trend=loss_trend,
                                                     acc=acc,
                                                     acc_trend=acc_trend)

    def on_validation_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
            The callback for the start of the validation.
        """
        self.interval = time.time()

    def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
            The callback for the end of the validation.

            Parameters
            ----------
            trainer : pl.Trainer
                The pyTorch trainer. Needed to get the loss and accuracy.
        """
        loss = trainer.logged_metrics["val_loss"].item()
        acc = trainer.logged_metrics["val_acc"].item()
        loss_trend = acc_trend = 0

        if trainer.current_epoch != 0:
            if self.comp_test_loss - loss != 0 and self.comp_test_loss != 0:
                loss_trend = int((loss - self.comp_test_loss) / self.comp_test_loss * 100)

            if self.comp_test_acc - acc != 0 and self.comp_test_acc != 0:
                acc_trend = int((acc - self.comp_test_acc) / self.comp_test_acc * 100)

        self.comp_test_loss = loss
        self.comp_test_acc = acc

        self.metrics["epoch"] = insert_test_summary(self.metrics["epoch"], loss=loss, acc=acc,
                                                    loss_trend=loss_trend, acc_trend=acc_trend)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
            The callback for the end of a validation batch.

            Parameters
            ----------
            trainer : pl.Trainer
                The pyTorch trainer. Needed to get the loss and accuracy.
        """
        if time.time() - self.interval > 1:
            self.memory = self.process.memory_info()[0] / (2 ** 20)
            self.meter.end()
            self.energy = self.meter.result.pkg[0] * 0.00000000028
            self.interval = time.time()
            self.meter.begin()

        loss = outputs["val_loss"].item()
        acc = outputs["val_acc"].item()
        time_test = time.perf_counter() - self.train_begin_time

        if batch_idx == 0:
            self.first_loss = loss
            self.first_acc = acc
        if self.first_loss - loss != 0:
            loss_trend = int((loss - self.first_loss)/self.first_loss * 100)
        else:
            loss_trend = 0

        if self.first_acc - acc != 0:
            acc_trend = int((acc - self.first_acc)/self.first_acc * 100)
        else:
            acc_trend = 0

        self.metrics["live"] = insert_live_metrics(metrics=self.metrics["live"],
                                                   phase="validation",
                                                   current_batch=batch_idx + 1,
                                                   loss=loss,
                                                   loss_trend=loss_trend,
                                                   acc=acc,
                                                   acc_trend=acc_trend,
                                                   memory=self.memory,
                                                   time=time_test,
                                                   energy=self.energy,
                                                   cpu=psutil.cpu_percent())

    def on_test_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
            The callback for the start of testing.
        """
        self.interval = time.time()

    def on_test_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
            The callback for the end of testing.

            Parameters
            ----------
            trainer : pl.Trainer
                The pyTorch trainer. Needed to get the loss and accuracy.
        """
        loss = trainer.logged_metrics["test_loss"].item()
        acc = trainer.logged_metrics["test_acc"].item()
        loss_trend = acc_trend = 0

        if trainer.current_epoch != 0:
            if self.comp_test_loss - loss != 0 and self.comp_test_loss != 0:
                loss_trend = int((loss - self.comp_test_loss) / self.comp_test_loss * 100)

            if self.comp_test_acc - acc != 0 and self.comp_test_acc != 0:
                acc_trend = int((acc - self.comp_test_acc) / self.comp_test_acc * 100)

        self.comp_test_loss = loss
        self.comp_test_acc = acc

        self.metrics["epoch"] = insert_test_summary(self.metrics["epoch"], loss=loss, acc=acc,
                                                    loss_trend=loss_trend, acc_trend=acc_trend)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
            The callback for the end of a testing batch.

            Parameters
            ----------
            trainer : pl.Trainer
                The pyTorch trainer. Needed to get the loss and accuracy.
        """
        if time.time() - self.interval > 1:
            self.memory = self.process.memory_info()[0] / (2 ** 20)
            self.meter.end()
            self.energy = self.meter.result.pkg[0] * 0.00000000028
            self.interval = time.time()
            self.meter.begin()

        loss = outputs["test_loss"].item()
        acc = outputs["test_acc"].item()
        time_test = time.perf_counter() - self.train_begin_time

        if batch_idx == 0:
            self.first_loss = loss
            self.first_acc = acc
        if self.first_loss - loss != 0:
            loss_trend = int((loss - self.first_loss)/self.first_loss * 100)
        else:
            loss_trend = 0

        if self.first_acc - acc != 0:
            acc_trend = int((acc - self.first_acc)/self.first_acc * 100)
        else:
            acc_trend = 0

        self.metrics["live"] = insert_live_metrics(metrics=self.metrics["live"],
                                                   phase="test",
                                                   current_batch=batch_idx + 1,
                                                   loss=loss,
                                                   loss_trend=loss_trend,
                                                   acc=acc,
                                                   acc_trend=acc_trend,
                                                   memory=self.memory,
                                                   time=time_test,
                                                   energy=self.energy,
                                                   cpu=psutil.cpu_percent())


class RtBenchTensorflow(callbacks.Callback):
    """
        The Tensorflow version of the main Benchmarking tool.

        Attributes
        ----------
        connection : db.Connection
            The connection to the database engine.
        logs : db.Table
            The database table for logging the metrics.
        run_begin_time : datetime
            The starttime of the run. Used to uniquely identify a run.
        first_loss : int
            The first loss of an epoch. Needed for trend comparison.
        first_acc : int
            The first accuracy of an epoch. Needed for trend comparison.
        train_begin_time : float
            The starttime of the training phase.
        epoch_begin_time : float
            The starttime of the epoch.
        test_begin_time : float
            The starttime of the test phase.
        interval : float
            Interval tracker for consistent memory and energy measuring.
        energy : float
            Amount of energy used.
        memory : float
            Amount of memory used.
        webapp : WebApp
            The corresponding webapp for visualization.
        comp_train_loss : float
            The training loss of the previous epoch. Needed for trend in the epoch summaries.
        comp_train_acc : float
            The training accuracy of the previous epoch. Needed for trend in the epoch summaries.
        comp_test_loss : float
            The testing loss of the previous epoch. Needed for trend in the epoch summaries.
        comp_test_acc : float
            The testing accuracy of the previous epoch. Needed for trend in the epoch summaries.
        no_batches : int
            The total number of batches.
        no_epochs : int
            The total number of epochs.
        metrics : dict
            All the key metrics to be shared between the webapp and RtBench.
        process : Process
            The psutil process for CPU and memory measurement.
        meter : Measurement
            The pyRAPL meter for energy measurement.
    """
    def __init__(self, input_size, batch_size, no_epochs):

        self.no_batches = ceil(input_size / batch_size)
        self.no_epochs = no_epochs
        pyRAPL.setup()
        self.meter = pyRAPL.Measurement('bar')

        self.run_begin_time = self.current_epoch = self.first_loss = self.first_acc = \
            self.train_begin_time = self.epoch_begin_time = self.test_begin_time = self.interval = \
            self.memory = self.energy = self.comp_train_loss = self.comp_train_acc = \
            self.comp_test_loss = self.comp_test_acc = 0

        engine = db.create_engine('sqlite:///logs.db')
        self.connection = engine.connect()
        metadata = db.MetaData()
        self.logs = setup_table(metadata)
        metadata.create_all(engine)

        manager = Manager()
        self.metrics = manager.dict()
        live_keys = ["epoch", "no_epochs", "batch", "no_batches", "loss", "loss_trend", "acc", "acc_trend",
                     "memory", "time", "eta_epoch", "eta_train", "energy", "cpu", "phase"]
        epoch_keys = ["epoch", "loss", "loss_trend", "acc", "acc_trend", "time", "avg_time", "test_loss", "test_acc",
                      "test_loss_trend", "test_acc_trend"]
        comparison_keys = ["time", "epoch", "batch", "loss", "loss_trend", "acc", "acc_trend",
                           "memory", "cpu", "energy", "available"]
        self.metrics["live"] = dict.fromkeys(live_keys)
        self.metrics["epoch"] = dict.fromkeys(epoch_keys)
        self.metrics["batch_update"] = 1
        self.metrics["no_batches"] = self.no_batches
        self.metrics["comparison_choice"] = ""
        self.metrics["comparison_options"] = self.get_comparison_options()
        self.metrics["comparison"] = dict.fromkeys(comparison_keys)
        self.metrics["comparison"]["available"] = False

        self.webApp = setup_webapp(metrics=self.metrics)
        self.process = psutil.Process(os.getpid())

    def get_comparison_options(self):
        """
            Returns all the available runs available for comparison.

            Returns
            -------
            results : datetime[]
                Array of all the available comparison runs in the database.
        """
        query = db.select([self.logs.columns.Starttime.distinct()])
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.fetchall()
        return [r[0] for r in ResultSet]

    def get_comparison(self):
        """
            Updates the comparison run depending on the selection and the current progress of the live run.
        """
        time_comp = time.perf_counter() - self.train_begin_time
        query = db.select([self.logs]).where(db.and_(self.logs.columns.Starttime ==
                                                     self.metrics["comparison_choice"],
                                                     self.logs.columns.Time > time_comp)).order_by(
                                                     db.asc(self.logs.columns.Time))
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.first()

        self.metrics["comparison"] = insert_comparison(self.metrics["comparison"], ResultSet)

    def on_train_begin(self, logs=None):
        """
            The callback for the start of training.

            Parameters
            ----------
            logs : dict
                Contains the metrics.
        """
        self.train_begin_time = time.perf_counter()
        self.run_begin_time = datetime.now().replace(microsecond=0).isoformat(' ')
        self.interval = time.time()
        self.meter.begin()

    def on_epoch_begin(self, epoch, logs=None):
        """
            The callback for the start of an epoch.

            Parameters
            ----------
            epoch : int
                The current epoch number.
            logs : dict
                Contains the metrics.
        """
        self.current_epoch = epoch
        self.epoch_begin_time = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        """
            The callback for the end of a batch.

            Parameters
            ----------
            batch : int
                The current batch number.
            logs : dict
                Contains the metrics.
        """
        if batch%self.metrics["batch_update"] != 0:
            return
        if time.time() - self.interval > 1:
            self.memory = self.process.memory_info()[0] / (2 ** 20)
            self.meter.end()
            if self.meter.result.pkg is not None:
                self.energy = self.meter.result.pkg[0] * 0.00000000028
            self.interval = time.time()
            self.meter.begin()

        self.get_comparison()
        time_epoch = time.perf_counter() - self.epoch_begin_time
        time_train = time.perf_counter() - self.train_begin_time
        eta_epoch = (time_epoch * self.no_batches / (batch+1)) - time_epoch
        eta_train = (time_train * (self.no_epochs * self.no_batches /
                           (batch + 1 + (self.current_epoch) * self.no_batches))) - time_train
        if batch == 0:
            self.first_loss = logs["loss"]
            self.first_acc = logs["accuracy"]
        if self.first_loss - logs["loss"] != 0:
            loss_trend = int((logs["loss"] - self.first_loss)/self.first_loss * 100)
        else:
            loss_trend = 0

        if self.first_acc - logs["accuracy"] != 0:
            acc_trend = int((logs["accuracy"] - self.first_acc)/self.first_acc * 100)
        else:
            acc_trend = 0

        query = db.insert(self.logs).values(Starttime=self.run_begin_time, Time="%.1f" % time_train, Epoch=self.current_epoch + 1,
                                            Batch=batch + 1, Loss="%.3f" % logs["loss"], LossTrend=loss_trend,
                                            Accuracy="%.3f" % logs["accuracy"], AccuracyTrend=acc_trend, Memory="%.1f" % self.memory,
                                            CPU=psutil.cpu_percent(),
                                            Power="%.4f" % self.energy)
        self.connection.execute(query)
        self.connection.execute(db.select([self.logs])).fetchall()

        self.metrics["live"] = insert_live_metrics(metrics=self.metrics["live"],
                                                   phase="train",
                                                   current_epoch=self.current_epoch + 1,
                                                   no_epochs=self.no_epochs,
                                                   current_batch=batch + 1,
                                                   no_batches=self.no_batches,
                                                   loss=logs["loss"],
                                                   loss_trend=loss_trend,
                                                   acc=logs["accuracy"],
                                                   acc_trend=acc_trend,
                                                   memory=self.memory,
                                                   time=time_train,
                                                   eta_epoch=eta_epoch,
                                                   eta_train=eta_train,
                                                   energy=self.energy,
                                                   cpu=psutil.cpu_percent())

    def on_epoch_end(self, epoch, logs=None):
        """
            The callback for the end of an epoch.

            Parameters
            ----------
            epoch : int
                The current epoch number.
            logs : dict
                Contains the metrics.
        """
        time_epoch = time.perf_counter() - self.epoch_begin_time

        loss = logs["loss"]
        acc = logs["accuracy"]
        loss_trend = acc_trend = 0

        if self.current_epoch != 0:
            if self.comp_train_loss - loss != 0 and self.comp_train_loss != 0:
                loss_trend = int((loss - self.comp_train_loss) / self.comp_train_loss * 100)
            if self.comp_train_acc - acc != 0 and self.comp_train_acc != 0:
                acc_trend = int((acc - self.comp_train_acc) / self.comp_train_acc * 100)

        self.comp_train_loss = loss
        self.comp_train_acc = acc

        self.metrics["epoch"] = insert_epoch_summary(metrics=self.metrics["epoch"],
                                                     current_epoch=epoch + 1,
                                                     time=time_epoch,
                                                     loss=loss,
                                                     avg_time=time_epoch / self.no_batches,
                                                     loss_trend=loss_trend,
                                                     acc=acc,
                                                     acc_trend=acc_trend)

    def on_test_begin(self, logs=None):
        """
            The callback for the start of testing.

            Parameters
            ----------
            logs : dict
                Contains the metrics.
        """
        self.interval = time.time()

    def on_test_batch_end(self, batch, logs=None):
        """
            The callback for the end of a test batch.

            Parameters
            ----------
            batch : int
                The current batch number.
            logs : dict
                Contains the metrics.
        """
        if time.time() - self.interval > 1:
            self.memory = self.process.memory_info()[0] / (2 ** 20)
            self.meter.end()
            self.energy = self.meter.result.pkg[0] * 0.00000000028
            self.interval = time.time()
            self.meter.begin()
        time_test = time.perf_counter() - self.train_begin_time

        if batch == 0:
            self.first_loss = logs["loss"]
            self.first_acc = logs["accuracy"]
        if self.first_loss - logs["loss"] != 0:
            loss_trend = int((logs["loss"] - self.first_loss)/self.first_loss * 100)
        else:
            loss_trend = 0

        if self.first_acc - logs["accuracy"] != 0:
            acc_trend = int((logs["accuracy"] - self.first_acc)/self.first_acc * 100)
        else:
            acc_trend = 0

        self.metrics["live"] = insert_live_metrics(metrics=self.metrics["live"],
                                                   phase="test",
                                                   current_batch=batch + 1,
                                                   loss=logs["loss"],
                                                   loss_trend=loss_trend,
                                                   acc=logs["accuracy"],
                                                   acc_trend=acc_trend,
                                                   memory=self.memory,
                                                   time=time_test,
                                                   energy=self.energy,
                                                   cpu=psutil.cpu_percent())

    def on_test_end(self, logs=None):
        """
            The callback for the end of testing.

            Parameters
            ----------
            logs : dict
                Contains the metrics.
        """
        loss = logs["loss"]
        acc = logs["accuracy"]
        loss_trend = acc_trend = 0

        if self.current_epoch != 0:
            if self.comp_test_loss - loss != 0 and self.comp_test_loss != 0:
                loss_trend = int((loss - self.comp_test_loss) / self.comp_test_loss * 100)

            if self.comp_test_acc - acc != 0 and self.comp_test_acc != 0:
                acc_trend = int((acc - self.comp_test_acc) / self.comp_test_acc * 100)

        self.comp_test_loss = loss
        self.comp_test_acc = acc

        self.metrics["epoch"] = insert_test_summary(self.metrics["epoch"], loss=logs["loss"], acc=logs["accuracy"],
                                                    loss_trend=loss_trend, acc_trend=acc_trend)
