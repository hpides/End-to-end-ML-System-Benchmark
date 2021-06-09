from tensorflow.keras import callbacks
import time
import psutil
import pyRAPL
import os
from math import ceil
import sqlalchemy as db
from datetime import datetime
from realTimeBench.webapp import WebApp
import multiprocessing
from pytorch_lightning.callbacks import Callback


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

        metadata.create_all(engine)  # Creates the table

        self.webApp = WebApp(self)
        server_process = multiprocessing.Process(target=self.webApp.run)
        server_process.start()
        time.sleep(0.5)

    first_loss = 0
    first_acc = 0

    process = psutil.Process(os.getpid())
    before_time = 0
    train_begin_time = 0
    epoch_begin_time = 0

    def updateComparison(self):
        f = open("comp.txt", "r")
        s = f.read()
        f.close()
        return s

    def getComparisonOptions(self):
        query = db.select([self.logs.columns.Starttime.distinct()])
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.fetchall()
        return [r[0] for r in ResultSet]

    def getComparison(self):
        f = open("compChoice.txt", "w")
        f.close()
        f = open("compChoice.txt", "r")
        s = f.read()
        f.close()

        time_comp = time.perf_counter() - self.train_begin_time
        #query = db.select([self.logs]).where(db.and_(self.logs.columns.Starttime == '2021-06-03 08:09:12.062038', self.logs.columns.Time > time_comp)).order_by(db.asc(self.logs.columns.Time))
        query = db.select([self.logs]).where(db.and_(self.logs.columns.Starttime == s.replace('T', ' '),
                                                     self.logs.columns.Time > time_comp)).order_by(
                                                     db.asc(self.logs.columns.Time))
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.first()

        f = open("comp.txt", "w")
        s = ""
        if ResultSet is not None:
            for i in range(len(ResultSet)-1):
                s+= " " + str(ResultSet[i+1])
        f.write(s)
        f.close()

    def update(self):
        f = open("temp.txt", "r")
        s = f.read()
        f.close()
        return s

    def updateEpoch(self):
        f = open("epoch.txt", "r")
        s = f.read()
        f.close()
        return s

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        f = open("epoch.txt", "w")
        f.write("--- --- --- --- --- --- ---")
        f.close()
        f = open("comp.txt", "w")
        f.write("--- --- --- --- --- --- --- --- --- ---")
        f.close()
        self.train_begin_time = time.perf_counter()
        self.starttime = datetime.now()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        #print(batch_idx)
        self.before_time = time.perf_counter()

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        #print(trainer.current_epoch)
        self.epoch_begin_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
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

        f = open("temp.txt", "w")
        f.write("{:2}/{}\t {:5}/{}\t  {:7.4f}, {}%\t {:8.4f}, {}% {:5.0f}MB \t\t\t{:4.0f}m{}s " \
                "\t\t\t{:4.0f}m{}s \t\t\t{:6.0f}m{}s\t\t\t{:6.5f}Wh \t\t{}%".format(
            current_epoch + 1, self.no_epochs, batch + 1, self.no_batches, loss, loss_trend,
            acc, acc_trend, memory,
            time_epoch // 60, int(time_epoch % 60), eta_epoch // 60, int(eta_epoch % 60), eta_train // 60,
            int(eta_train % 60),
            energy, psutil.cpu_percent()))
        f.close()

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        loss = trainer.logged_metrics["train_loss"].item()
        acc = trainer.logged_metrics["train_acc"].item()
        self.getComparison()
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

        f = open("epoch.txt", "w")
        f.write(
            "{} {:2.0f}m{}s {:4.2f}s {:6.4f} {}% {:6.4f} {}%".format(trainer.current_epoch + 1, time_epoch // 60, int(time_epoch % 60),
                                                                     time_epoch / self.no_batches, loss,
                                                                     loss_trend,
                                                                     acc, acc_trend))
        f.close()




class realTimeBenchmarker(callbacks.Callback):

    def __init__(self, input_size, batch_size, no_epochs):

        self.no_batches = ceil(input_size / batch_size)
        self.no_epochs = no_epochs
        pyRAPL.setup()
        self.meter = pyRAPL.Measurement('bar')

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

        metadata.create_all(engine)  # Creates the table

        self.webApp = WebApp(self)
        server_process = multiprocessing.Process(target=self.webApp.run)
        server_process.start()
        time.sleep(0.5)

    current_epoch = 5
    first_loss = 0
    first_acc = 0

    process = psutil.Process(os.getpid())
    before_time = 0
    train_begin_time = 0
    epoch_begin_time = 0

    def updateComparison(self):
        f = open("compChoice.txt", "w")
        f.close()
        f = open("comp.txt", "r")
        s = f.read()
        f.close()
        return s

    def getComparisonOptions(self):
        query = db.select([self.logs.columns.Starttime.distinct()])
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.fetchall()
        return [r[0] for r in ResultSet]

    def getComparison(self):
        f = open("compChoice.txt", "r")
        s = f.read()
        f.close()

        time_comp = time.perf_counter() - self.train_begin_time
        #query = db.select([self.logs]).where(db.and_(self.logs.columns.Starttime == '2021-06-03 08:09:12.062038', self.logs.columns.Time > time_comp)).order_by(db.asc(self.logs.columns.Time))
        query = db.select([self.logs]).where(db.and_(self.logs.columns.Starttime == s.replace('T', ' '),
                                                     self.logs.columns.Time > time_comp)).order_by(
                                                     db.asc(self.logs.columns.Time))
        ResultProxy = self.connection.execute(query)
        ResultSet = ResultProxy.first()

        f = open("comp.txt", "w")
        s = ""
        if ResultSet is not None:
            for i in range(len(ResultSet)-1):
                s+= " " + str(ResultSet[i+1])
        f.write(s)
        f.close()

    def update(self):
        f = open("temp.txt", "r")
        s = f.read()
        f.close()
        return s

    def updateEpoch(self):
        f = open("epoch.txt", "r")
        s = f.read()
        f.close()
        return s

    def on_train_begin(self, logs=None):
        f = open("epoch.txt", "w")
        f.write("--- --- --- --- --- --- ---")
        f.close()
        f = open("comp.txt", "w")
        f.write("--- --- --- --- --- --- --- --- --- ---")
        f.close()
        self.train_begin_time = time.perf_counter()
        self.starttime = datetime.now()

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.epoch_begin_time = time.perf_counter()

    def on_batch_begin(self, batch, logs=None):
        self.meter.begin()
        self.before_time = time.perf_counter()

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
            if loss_trend > 0:
                loss_trend = "+" + str(loss_trend)

        else:
            loss_trend = 0

        if self.first_acc - logs["accuracy"] != 0:
            acc_trend = int((logs["accuracy"] - self.first_acc)/self.first_acc * 100)
            if acc_trend > 0:
                acc_trend = "+" + str(acc_trend)
        else:
            acc_trend = 0

        query = db.insert(self.logs).values(Starttime=self.starttime, Time="%.1f" % time_train, Epoch=self.current_epoch + 1,
                                            Batch=batch + 1, Loss="%.3f" % logs["loss"], LossTrend=loss_trend,
                                            Accuracy="%.3f" % logs["accuracy"], AccuracyTrend=acc_trend, Memory="%.1f" %memory,
                                            CPU=psutil.cpu_percent(),
                                            Power="%.4f" % (self.meter.result.pkg[0] * 0.00000000028))
        ResultProxy = self.connection.execute(query)
        results = self.connection.execute(db.select([self.logs])).fetchall()

        f = open("temp.txt", "w")
        f.write("{:2}/{}\t {:5}/{}\t  {:7.4f}, {}%\t {:8.4f}, {}% {:5.0f}MB \t\t\t{:4.0f}m{}s " \
                                   "\t\t\t{:4.0f}m{}s \t\t\t{:6.0f}m{}s\t\t\t{:6.5f}Wh \t\t{}%".format(
            self.current_epoch + 1, self.no_epochs, batch + 1, self.no_batches, logs["loss"], loss_trend, logs["accuracy"], acc_trend, memory,
            time_epoch//60, int(time_epoch%60), eta_epoch//60,int(eta_epoch%60), eta_train//60, int(eta_train%60),
            self.meter.result.pkg[0] * 0.00000000028, psutil.cpu_percent()))
        f.close()

    def on_epoch_end(self, epoch, logs=None):
        self.getComparison()
        print(self.model.get_weights())
        time_epoch = time.perf_counter() - self.epoch_begin_time
        if self.first_loss - logs["loss"] != 0:
            loss_trend = int((logs["loss"] - self.first_loss) / self.first_loss * 100)
            if loss_trend > 0:
                loss_trend = "+" + str(loss_trend)
        else:
            loss_trend = 0

        if self.first_acc - logs["accuracy"] != 0:
            acc_trend = int((logs["accuracy"] - self.first_acc) / self.first_acc * 100)
            if acc_trend > 0:
                acc_trend = "+" + str(acc_trend)
        else:
            acc_trend = 0

        f = open("epoch.txt", "w")
        f.write("{} {:2.0f}m{}s {:4.2f}s {:6.4f} {}% {:6.4f} {}%".format(epoch + 1, time_epoch//60, int(time_epoch%60),
                                                  time_epoch / self.no_batches, logs["loss"], loss_trend,
                                                                        logs["accuracy"], acc_trend))
        f.close()
