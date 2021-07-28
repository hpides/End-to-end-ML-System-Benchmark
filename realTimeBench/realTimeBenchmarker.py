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
    metrics["epoch"] = current_epoch
    metrics["time"] = time
    metrics["avg_time"] = avg_time
    metrics["loss"] = loss
    metrics["loss_trend"] = loss_trend
    metrics["acc"] = acc
    metrics["acc_trend"] = acc_trend
    return metrics


def insert_test_summary(metrics, loss, acc, loss_trend, acc_trend):
    metrics["test_loss"] = loss
    metrics["test_acc"] = acc
    metrics["test_loss_trend"] = loss_trend
    metrics["test_acc_trend"] = acc_trend
    return metrics


def insert_comparison(metrics, ResultSet):
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
    webApp = WebApp()
    server_process = Process(target=webApp.run, args=(metrics,))
    server_process.start()
    time.sleep(0.5)
    return webApp


def track(memory_values):
    process = psutil.Process(os.getpid())
    pyRAPL.setup()
    meter = pyRAPL.Measurement('bar')
    meter.begin()
    starttime = time.time()
    while True:
        time.sleep(10.0 - ((time.time() - starttime) % 10.0))
        memory_values["memory"] = process.memory_info()[0] / (2 ** 20)
        meter.end()
        memory_values["energy"] = meter.result.pkg[0] * 0.00000000028
        meter.begin()


def setup_parallel_trackers(memory_values):
    server_process = Process(target=track, args=(memory_values,))
    server_process.start()
    time.sleep(0.5)


def setup_table(metadata):
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

    def __init__(self):
        engine = db.create_engine('sqlite:///logs.db?check_same_thread=False')
        self.connection = engine.connect()
        metadata = db.MetaData()
        self.logs = setup_table(metadata)
        metadata.create_all(engine)

        self.starttime = self.first_loss = self.first_acc = \
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
        query = db.select([self.logs.columns.Starttime.distinct()])
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.fetchall()
        return [r[0] for r in ResultSet]

    def get_comparison(self):
        time_comp = time.perf_counter() - self.train_begin_time
        query = db.select([self.logs]).where(db.and_(self.logs.columns.Starttime ==
                                                     self.metrics["comparison_choice"],
                                                     self.logs.columns.Time > time_comp)).order_by(
            db.asc(self.logs.columns.Time))
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.first()

        self.metrics["comparison"] = insert_comparison(self.metrics["comparison"], ResultSet)

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.no_epochs = self.metrics["no_epochs"] = trainer.max_epochs
        self.no_batches = self.metrics["no_batches"] = trainer.num_training_batches + trainer.num_val_batches[0]
        self.webApp = setup_webapp(metrics=self.metrics)
        self.train_begin_time = time.perf_counter()
        self.starttime = datetime.now().replace(microsecond=0).isoformat(' ')
        self.interval = time.time()
        self.meter.begin()

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.epoch_begin_time = time.perf_counter()

    #def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
    #    self.meter.begin()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
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

        query = db.insert(self.logs).values(Starttime=self.starttime, Time="%.1f" % time_train, Epoch=trainer.current_epoch + 1,
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
        #print(self.model.get_weights())
        print(trainer.logged_metrics)
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
        self.interval = time.time()

    def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
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

    def __init__(self, input_size, batch_size, no_epochs):

        self.no_batches = ceil(input_size / batch_size)
        self.no_epochs = no_epochs
        pyRAPL.setup()
        self.meter = pyRAPL.Measurement('bar')

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

        self.starttime = self.current_epoch = self.first_loss = self.first_acc = \
            self.train_begin_time = self.epoch_begin_time = self.test_begin_time = self.interval = \
            self.memory = self.energy = self.comp_train_loss = self.comp_train_acc = \
            self.comp_test_loss = self.comp_test_acc = 0
        self.process = psutil.Process(os.getpid())

    def get_comparison_options(self):
        query = db.select([self.logs.columns.Starttime.distinct()])
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.fetchall()
        return [r[0] for r in ResultSet]

    def get_comparison(self):
        time_comp = time.perf_counter() - self.train_begin_time
        query = db.select([self.logs]).where(db.and_(self.logs.columns.Starttime ==
                                                     self.metrics["comparison_choice"],
                                                     self.logs.columns.Time > time_comp)).order_by(
                                                     db.asc(self.logs.columns.Time))
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.first()

        self.metrics["comparison"] = insert_comparison(self.metrics["comparison"], ResultSet)

    def on_train_begin(self, logs=None):
        self.train_begin_time = time.perf_counter()
        self.starttime = datetime.now().replace(microsecond=0).isoformat(' ')
        self.interval = time.time()
        self.meter.begin()

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.epoch_begin_time = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
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

        query = db.insert(self.logs).values(Starttime=self.starttime, Time="%.1f" % time_train, Epoch=self.current_epoch + 1,
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
        self.interval = time.time()

    def on_test_batch_end(self, batch, logs=None):
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
        print(logs)
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
