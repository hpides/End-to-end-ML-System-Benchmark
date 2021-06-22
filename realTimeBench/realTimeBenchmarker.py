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


def setupWebapp(rtb, metrics):
    webApp = WebApp(rtb)
    server_process = Process(target=webApp.run, args=(metrics,))
    server_process.start()
    time.sleep(0.5)
    return webApp


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


class realTimeBenchmarker_pyTorch(Callback):

    def __init__(self, no_epochs, input_size, batch_size):
        pyRAPL.setup()
        self.meter = pyRAPL.Measurement('bar')
        self.no_batches = ceil(input_size / batch_size)
        self.no_epochs = no_epochs

        engine = db.create_engine('sqlite:///logs.db')  # Create
        self.connection = engine.connect()
        metadata = db.MetaData()

        self.logs = db.Table('logs', metadata,
                             db.Column('Starttime', db.DateTime()),
                             db.Column('Time', db.Float()),
                             db.Column('Epoch', db.Integer()),
                             db.Column('Batch', db.Integer()),
                             db.Column('Loss', db.Float()),
                             db.Column('LossTrend', db.Float()),
                             db.Column('Accuracy', db.Float()),
                             db.Column('AccuracyTrend', db.Float()),
                             db.Column('Memory', db.Float()),
                             db.Column('CPU', db.Float()),
                             db.Column('Power', db.Float())
                             )

        metadata.create_all(engine)

        manager = Manager()
        self.metrics = manager.dict()
        live_keys = ["epoch", "no_epochs", "batch", "no_batches", "loss", "loss_trend", "acc", "acc_trend",
                     "memory", "time", "eta_epoch", "eta_train", "energy", "cpu"]
        self.metrics["live"] = dict.fromkeys(live_keys)
        self.metrics["epoch"] = ["---", "---", "---", "---", "---", "---", "---"]
        self.metrics["comparison_choice"] = ""
        self.metrics["comparison"] = ["---", "---", "---", "---", "---", "---", "---", "---", "---", "---"]

        self.webApp = WebApp(self)
        server_process = Process(target=self.webApp.run, args=(self.metrics,))
        server_process.start()
        time.sleep(0.5)

        self.first_loss = 0
        self.first_acc = 0
        self.process = psutil.Process(os.getpid())
        self.train_begin_time = 0
        self.epoch_begin_time = 0

    def getComparisonOptions(self):
        query = db.select([self.logs.columns.Starttime.distinct()])
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.fetchall()
        return [r[0] for r in ResultSet]

    def getComparison(self):
        s = self.metrics["comparison_choice"]

        time_comp = time.perf_counter() - self.train_begin_time
        # query = db.select([self.logs]).where(db.and_(self.logs.columns.Starttime == '2021-06-03 08:09:12.062038', self.logs.columns.Time > time_comp)).order_by(db.asc(self.logs.columns.Time))
        query = db.select([self.logs]).where(db.and_(self.logs.columns.Starttime == s.replace('T', ' '),
                                                     self.logs.columns.Time > time_comp)).order_by(
            db.asc(self.logs.columns.Time))
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.first()

        results = []

        if ResultSet is not None:
            for i in range(len(ResultSet) - 1):
                results.append(str(ResultSet[i + 1]))
            self.metrics["comparison"] = results
        else:
            self.metrics["comparison"] = ["---", "---", "---", "---", "---", "---", "---", "---", "---", "---"]


    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.train_begin_time = time.perf_counter()
        self.starttime = datetime.now()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.webApp.test = batch_idx

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.epoch_begin_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 20 != 0:
            return
        self.getComparison()
        loss = outputs["loss"].item()
        acc = outputs["acc"].item()
        batch=batch_idx
        current_epoch = trainer.current_epoch
        #self.meter.end()
        memory = self.process.memory_info()[0] / (2 ** 20)
        time_epoch = time.perf_counter() - self.epoch_begin_time
        time_train = time.perf_counter() - self.train_begin_time
        eta_epoch = (time_epoch * self.no_batches / (batch + 1)) - time_epoch
        eta_train = (time_train * (self.no_epochs * self.no_batches /
                                   (batch + 1 + (current_epoch) * self.no_batches))) - time_train
        if batch == 0:
            self.first_loss = loss
            self.first_acc = acc
        if self.first_loss - loss != 0:
            loss_trend = int((loss - self.first_loss) / self.first_loss * 100)
            if loss_trend > 0:
                loss_trend = "+" + str(loss_trend)
        else:
            loss_trend = 0

        if self.first_acc - acc != 0:
            acc_trend = int((acc - self.first_acc)/self.first_acc * 100)
            if acc_trend > 0:
                acc_trend = "+" + str(acc_trend)
        else:
            acc_trend = 0

        energy = 0.00054

        query = db.insert(self.logs).values(Starttime=self.starttime, Time="%.1f" % time_train,
                                            Epoch=current_epoch + 1,
                                            Batch=batch + 1, Loss="%.3f" % loss, LossTrend=loss_trend,
                                            Accuracy="%.3f" % acc, AccuracyTrend=acc_trend,
                                            Memory="%.1f" % memory,
                                            CPU=psutil.cpu_percent(),
                                            Power="%.4f" % (energy))
        ResultProxy = self.connection.execute(query)
        results = self.connection.execute(db.select([self.logs])).fetchall()

        print(self.metrics["live"])

        self.metrics["live"] = [
            "{:2}/{}".format(current_epoch + 1, self.no_epochs),
            "{:5}/{}".format(batch + 1, self.no_batches),
            "{:7.4f}".format(loss),
            "{}%".format(loss_trend),
            "{:8.4f}".format(acc),
            "{}%".format(acc_trend),
            "{:5.0f}MB ".format(memory),
            "{:4.0f}m{}s".format(time_epoch//60, int(time_epoch%60)),
            "{:4.0f}m{}s".format(eta_epoch//60,int(eta_epoch%60)),
            "{:6.0f}m{}s".format(eta_train//60, int(eta_train%60)),
            "{:6.5f}Wh".format(self.meter.result.pkg[0] * 0.00000000028),
            "{}%".format(psutil.cpu_percent())
        ]

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        loss = trainer.logged_metrics["train_loss"].item()
        acc = trainer.logged_metrics["train_acc"].item()
        time_epoch = time.perf_counter() - self.epoch_begin_time
        if self.first_loss - loss != 0:
            loss_trend = int((loss - self.first_loss) / self.first_loss * 100)
            if loss_trend > 0:
                loss_trend = "+" + str(loss_trend)
        else:
            loss_trend = 0

        if self.first_acc - acc != 0:
            acc_trend = int((acc - self.first_acc) / self.first_acc * 100)
            if acc_trend > 0:
                acc_trend = "+" + str(acc_trend)
        else:
            acc_trend = 0

        self.metrics["epoch"] = [
            "{}".format(trainer.current_epoch + 1),
            "{:2.0f}m{}s".format(time_epoch//60, int(time_epoch%60)),
            "{:4.2f}s".format(time_epoch / self.no_batches),
            "{:6.4f}".format(loss),
            "{}%".format(loss_trend),
            "{:6.4f}".format(acc),
            "{}%".format(acc_trend)
        ]


class realTimeBenchmarker_tensorflow(callbacks.Callback):

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
        epoch_keys = ["epoch", "loss", "loss_trend", "acc", "acc_trend", "time", "avg_time"]
        comparison_keys = ["time", "epoch", "batch", "loss", "loss_trend", "acc", "acc_trend",
                           "memory", "cpu", "energy", "available"]
        self.metrics["live"] = dict.fromkeys(live_keys)
        self.metrics["epoch"] = dict.fromkeys(epoch_keys)
        self.metrics["comparison_choice"] = ""
        self.metrics["comparison"] = dict.fromkeys(comparison_keys)
        self.metrics["comparison"]["available"] = False

        self.webApp = setupWebapp(rtb=self, metrics=self.metrics)

        self.starttime = self.current_epoch = self.first_loss = self.first_acc = \
            self.train_begin_time = self.epoch_begin_time = self.test_begin_time = 0
        self.process = psutil.Process(os.getpid())

    def getComparisonOptions(self):
        query = db.select([self.logs.columns.Starttime.distinct()])
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.fetchall()
        return [r[0] for r in ResultSet]

    def getComparison(self):
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

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.epoch_begin_time = time.perf_counter()

    def on_batch_begin(self, batch, logs=None):
        self.meter.begin()

    def on_train_batch_end(self, batch, logs=None):
        self.getComparison()
        self.meter.end()
        memory = self.process.memory_info()[0] / (2 ** 20)
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
                                            Accuracy="%.3f" % logs["accuracy"], AccuracyTrend=acc_trend, Memory="%.1f" %memory,
                                            CPU=psutil.cpu_percent(),
                                            Power="%.4f" % (self.meter.result.pkg[0] * 0.00000000028))
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
                                                   memory=memory,
                                                   time=time_epoch,
                                                   eta_epoch=eta_epoch,
                                                   eta_train=eta_train,
                                                   energy=self.meter.result.pkg[0] * 0.00000000028,
                                                   cpu=psutil.cpu_percent())

    def on_epoch_end(self, epoch, logs=None):
        #print(self.model.get_weights())
        time_epoch = time.perf_counter() - self.epoch_begin_time
        if self.first_loss - logs["loss"] != 0:
            loss_trend = int((logs["loss"] - self.first_loss) / self.first_loss * 100)
        else:
            loss_trend = 0

        if self.first_acc - logs["accuracy"] != 0:
            acc_trend = int((logs["accuracy"] - self.first_acc) / self.first_acc * 100)
        else:
            acc_trend = 0

        print(epoch)

        self.metrics["epoch"] = insert_epoch_summary(metrics=self.metrics["epoch"],
                                                     current_epoch=epoch + 1,
                                                     time=time_epoch,
                                                     loss=logs["loss"],
                                                     avg_time=time_epoch / self.no_batches,
                                                     loss_trend=loss_trend,
                                                     acc=logs["accuracy"],
                                                     acc_trend=acc_trend)

    def on_test_begin(self, logs=None):
        self.test_begin_time = time.perf_counter()

    def on_test_batch_begin(self, batch, logs=None):
        self.meter.begin()

    def on_test_batch_end(self, batch, logs=None):
        self.meter.end()
        memory = self.process.memory_info()[0] / (2 ** 20)
        time_test = time.perf_counter() - self.test_begin_time

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
                                                   memory=memory,
                                                   time=time_test,
                                                   energy=self.meter.result.pkg[0] * 0.00000000028,
                                                   cpu=psutil.cpu_percent())
