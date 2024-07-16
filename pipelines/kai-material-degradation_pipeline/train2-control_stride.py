from model_2 import SimpleCNN
from dataset import WaveformsDataset

import torch
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from torchmetrics.functional import mean_absolute_error, mean_squared_error, r2_score


import os
import umlaut
import subprocess
import math
from tqdm import tqdm
from pprint import pprint

print("finished importing")


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128"

metrics = []
types = []

metrics.append(umlaut.MemoryMetric('memory', interval=0.1))
types.append("memory")

metrics.append(umlaut.GPUMemoryMetric('gpumemory', interval=0.1))
types.append("gpumemory")

metrics.append(umlaut.CPUMetric('cpu', interval=0.1))
types.append("cpu")

metrics.append(umlaut.GPUMetric('gpu', interval=0.1))
types.append("gpu")

metrics.append(umlaut.GPUPowerMetric('gpupower', interval=0.1))
types.append("gpupower")

metrics.append(umlaut.GPUTimeMetric('gputime'))
types.append("gputime")

metrics.append(umlaut.TimeMetric('time'))
types.append("time")

bm = umlaut.Benchmark('material-degradation.db', description="Benchmark KAI material degradation.")

# base_dir = r"../../data/processed/file_per_cycle_V1.1/TP1"
base_dir = "TP1/"
# base_dir = r"../../data/processed/file_per_cycle_V1.3_quadratic_and_normalized/TP1"

def prepare_windows(test_run_dir:str, number_of_labels:int, window_size:int)-> list[tuple[str, str, float, int]]:
    """
    Takes the path to the test run folder containing file per cycle, number of intended labels, and window size.
    returns a list of tuples of the start cycle and end cycle of each window, and the window label both for regression and classification
    """
    
    windows_list = []
    temp_df = pd.read_csv(os.path.join(test_run_dir, "cycles_metadata.csv")).drop('Unnamed: 0', axis= 1)
    
    cycles = temp_df.shape[0]
    stride = 5
    #no_of_windows = cycles - window_size + 1
    no_of_windows = int(((cycles - window_size)/stride) + 1)
    dut_name = test_run_dir.split('/')[-1].split("\\")[-1]

    for i in range(no_of_windows):
        first = dut_name + "_" + str(temp_df.recorded_cycle[i*stride])
        last = dut_name + "_" + str(temp_df.recorded_cycle[i*stride + window_size -1])
#         first = dut_name + "_" + str(temp_df.recorded_cycle[i])
#         last = dut_name + "_" + str(temp_df.recorded_cycle[i + window_size -1])
        regression_label = temp_df.label_for_regression [i*stride + window_size -1] #regression label index should be for the last cycle of the window
            
        window_classification_label = temp_df.label_for_regression[i*stride + window_size -1]*number_of_labels #classification label index should be for the last cycle of the window
        window_classification_label = math.floor(window_classification_label)
        windows_list.append((first, last, regression_label, window_classification_label))
    
    return windows_list

@umlaut.BenchmarkSupervisor(metrics, bm, name="awsome cusom name")
def make_metadata(base_dir:str, window_size:int, num_of_labels:int)-> list[tuple[str, str, float, int]]:
    """
    Runs prepare_windows over all files
    Takes the path to the base_dir with test run folders, window size, number of intended labels.
    Returns a list of tuples (for the whole dataset) of the start cycle and end cycle of each window, and 
    the window label both for regression and classification
    """
    
    dataset_metadata = []
      
    for i in tqdm(os.listdir(base_dir)):
        if i[-3:] != 'csv':
            dut_folder = os.path.join(base_dir, i)
            windows_list = prepare_windows(dut_folder, num_of_labels, window_size)
            dataset_metadata.extend(windows_list)

    return dataset_metadata


# parameters
num_of_labels = 3
window_size = 50
RANDOM_STATE = 48151623
k_folds = 5
#patience_epochs = 10

# intitalize Kfold 
kfold = KFold(n_splits=k_folds, shuffle=True, random_state = RANDOM_STATE)

# prepare dataset metadata (split dataset into windows of size <window_size>)
dataset_meta = make_metadata(base_dir, window_size, num_of_labels)
dataset = WaveformsDataset(base_dir, dataset_meta)


# more parameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
NUM_WORKERS = 64 # 1 if using 1 cpu, https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/2
# https://github.com/pytorch/pytorch/issues/5301#issuecomment-654265873
SHUFFLE = True
EPOCHS = 3


device = ("cuda" if torch.cuda.is_available() else "cpu") 
print("cuda is available ✅" if device == "cuda" else "cuda not available ❌❌")


def memory_stats():
    print(f"allocatedMem:  {torch.cuda.memory_allocated()/1024**2}, cachedMem:  {torch.cuda.memory_reserved()/1024**2}")
memory_stats()


# def train():

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    LEARNING_RATE = 5e-5

    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    if fold > 0:
        print(f"skipping fold {fold}..")
        continue

    eval_metrics = {"training_mse": [],
                    "training_rmse": [],
                    "validation_mse": [],
                    "validation_rmse": [],
                    }
    last_loss = 1

    # split train into train and validation
    trn_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=RANDOM_STATE)

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(trn_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=BATCH_SIZE, sampler=train_subsampler, num_workers=NUM_WORKERS)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=BATCH_SIZE, sampler=test_subsampler, num_workers=NUM_WORKERS)
    valloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=BATCH_SIZE, sampler=val_subsampler, num_workers=NUM_WORKERS)
    
    # model initialization
    loss_fn = torch.nn.MSELoss()
    model = SimpleCNN(window_size)
    model = model.to(device) #.half()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)
#     #weight initialization
#     torch.nn.init.xavier_uniform(model.conv1.weight)
#     model.conv1.bias.data.zero_()


    for epoch in range(EPOCHS):
        # Train
#             if epoch >= 30:
#                 LEARNING_RATE = 1e-5
#             if epoch >= 80:
#                 LEARNING_RATE = 5e-6

        @umlaut.BenchmarkSupervisor(metrics, bm)
        def train_step():
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            print(f"current learning rate {LEARNING_RATE}")
            
            total_loss = 0
            for i, (X,y) in tqdm(enumerate(trainloader)):
                X = X.to(device)
                y = y.to(device)
                X = X.float()
                y = y.float()

                model.train() # set the model in training mode
                outputs = model(X) 
                outputs = outputs.float() 
                loss = loss_fn(outputs,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() # update weights

                total_loss += loss.item()

            print(f"Epoch: {epoch}, Training MSE: {total_loss/len(trainloader)}, RMSE: {math.sqrt(total_loss/len(trainloader))}")

            eval_metrics["training_mse"].append(total_loss/len(trainloader))
            eval_metrics["training_rmse"].append(math.sqrt(total_loss/len(trainloader)))

        train_step()

        @umlaut.BenchmarkSupervisor(metrics, bm)
        def val_step(last_loss):
            # Validation
            val_loss = 0
            for i_val, (X, y) in tqdm(enumerate(valloader)):
                X = X.to(device)
                y = y.to(device)
                model.eval() # set the model in evaluation mode (deactivates BN and dropout laeyers)
                with torch.no_grad():
                    outputs_val = model(X)
                    val_loss += loss_fn(outputs_val,y).item()

            if val_loss/len(valloader) > last_loss:
                print(f"Validation MSE: {val_loss/len(valloader)}, RMSE: {math.sqrt(val_loss/len(valloader))}")


            eval_metrics["validation_mse"].append(val_loss/len(valloader))
            eval_metrics["validation_rmse"].append(math.sqrt(val_loss/len(valloader)))
            print(f"Validation MSE: {val_loss/len(valloader)}, RMSE: {math.sqrt(val_loss/len(valloader))}")

    #             pprint(eval_metrics)
            

            if epoch == EPOCHS - 1:
                pprint(eval_metrics)

    #                 if val_loss/len(valloader) > last_loss:
            if (epoch-1)%50 == 0 :
                mae_sum = 0
                mse_sum = 0
                r2_sum = 0
    #                     errors = []
                for i, (X, y) in tqdm(enumerate(testloader)):
                    X = X.to(device)
                    y = y.to(device)

                    X = X.float()
                    y = y.float()
                    model.eval()
                    with torch.no_grad():
                        outputs = model(X)

                    mae_sum += mean_absolute_error(outputs,y).item()
    #                         errors.extend(abs(outputs-y).tolist())
                    mse_sum += mean_squared_error(outputs,y).item()
                    r2_sum += r2_score(outputs,y).item()

                print(f"**** Testing Metrics - Epoch: {epoch}")
                print(f"Test MSE: {mse_sum/len(testloader)}")
                print(f"Test RMSE: {math.sqrt(mse_sum/len(testloader))}")
                print(f"Test MAE: {mae_sum/len(testloader)}")
                print(f"Test R2: {r2_sum/len(testloader)}")


                pprint(eval_metrics)
    #    
            last_loss = val_loss/len(valloader)
            return last_loss

        last_loss = val_step(last_loss)
        
    @umlaut.BenchmarkSupervisor(metrics, bm)
    def test_step():
        # Model Testing
        print(f"--------------------------------")
        print('********* testing *********')

        mae_sum = 0
        mse_sum = 0
        r2_sum = 0
        errors = []
        for i, (X, y) in tqdm(enumerate(testloader)):
            X = X.to(device)
            y = y.to(device)

            X = X.float()
            y = y.float()
            model.eval()
            with torch.no_grad():
                outputs = model(X)

            mae_sum += mean_absolute_error(outputs,y).item()
            errors.extend(abs(outputs-y).tolist())
            mse_sum += mean_squared_error(outputs,y).item()
            r2_sum += r2_score(outputs,y).item()


        print(f"Test MSE: {mse_sum/len(testloader)}")
        print(f"Test RMSE: {math.sqrt(mse_sum/len(testloader))}")
        print(f"Test MAE: {mae_sum/len(testloader)}")
        print(f"Test R2: {r2_sum/len(testloader)}")
        print(f"--------------------------------")

    test_step()

uuid = bm.uuid
print("UUID", uuid)
bm.close()

subprocess.run(["umlaut-cli", "material-degradation.db", "-u", uuid, "-t"] + types + ["-d"] + types + ["-p", "plotly"])

# if __name__ == '__main__':
#     train()
