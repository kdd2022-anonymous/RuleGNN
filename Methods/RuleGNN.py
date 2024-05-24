import os
from typing import List

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from GraphData import GraphData
from NeuralNetArchitectures import GraphNN
from utils.Parameters import Parameters
from Time.TimeClass import TimeClass
import TrainTestData.TrainTestData as ttd
from utils.utils import get_k_lowest_nonzero_indices


class RuleGNN:
    def __init__(self, run_id: int, k_val: int, graph_data: GraphData.GraphData, training_data: List[int],
                 validate_data: List[int], test_data: List[int], seed: int, para: Parameters.Parameters):
        self.run_id = run_id
        self.k_val = k_val
        self.graph_data = graph_data
        self.training_data = training_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.seed = seed
        self.para = para
        self.results_path = para.configs['paths']['results']

    def Run(self):
        torch.set_num_threads(1)
        dtype = torch.float
        if 'precision' in self.para.configs:
            if self.para.configs['precision'] == 'double':
                dtype = torch.double
                # set the inputs in graph_data to double precision
                self.graph_data.inputs = [x.double() for x in self.graph_data.inputs]
        """
        Set up the network
        """

        net = GraphNN.GraphNet(graph_data=self.graph_data,
                               para=self.para,
                               seed=self.seed)

        # get gpu or cpu: not used at the moment
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        timer = TimeClass()

        """
        Set up the loss function
        """
        if self.para.run_config.loss == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        if self.para.run_config.loss == 'MeanSquaredError':
            criterion = nn.MSELoss()
        if self.para.run_config.loss == 'MeanAbsoluteError':
            criterion = nn.L1Loss()

        """
        Set up the optimizer
        """
        optimizer = optim.Adam(net.parameters(), lr=self.para.learning_rate)

        if self.run_id == 0 and self.k_val == 0:
            # create a file about the net details including (net, optimizer, learning rate, loss function, batch size, number of classes, number of epochs, balanced data, dropout)
            file_name = f'{self.para.db}_{self.para.new_file_index}_Network.txt'
            file_name = f'{self.para.db}_{self.para.config_id}_Network.txt'
            with open(f'{self.results_path}{self.para.db}/Results/{file_name}', "a") as file_obj:
                file_obj.write(f"Network architecture: {self.para.run_config.network_architecture}\n"
                               f"Optimizer: {optimizer}\n"
                               f"Loss function: {criterion}\n"
                               f"Batch size: {self.para.batch_size}\n"
                               f"Balanced data: {self.para.balance_data}\n"
                               f"Number of epochs: {self.para.n_epochs}\n")
                # iterate over the layers of the neural net
                for layer in net.net_layers:
                    # get number of trainable parameters
                    layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                    file_obj.write(f"Trainable Parameters: {layer_params}\n")
                    try:
                        layer.node_labels
                        file_obj.write(f"Node labels: {layer.node_labels.num_unique_node_labels}\n")
                    except:
                        pass
                    try:
                        layer.edge_labels
                        file_obj.write(f"Edge labels: {layer.edge_labels.num_unique_edge_labels}\n")
                    except:
                        pass
                for name, param in net.named_parameters():
                    file_obj.write(f"Layer: {name} -> {param.requires_grad}\n")

        file_name = f'{self.para.db}_{self.para.new_file_index}_Results_run_id_{self.run_id}_validation_step_{self.para.validation_id}.csv'
        file_name = f'{self.para.db}_{self.para.config_id}_Results_run_id_{self.run_id}_validation_step_{self.para.validation_id}.csv'

        # header use semicolon as delimiter
        if self.graph_data.num_classes == 1 or self.para.run_config.task == 'regression':
            header = "Dataset;RunNumber;ValidationNumber;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss;EpochAccuracy;" \
                     "EpochMAE;EpochMAEStd;EpochTime;ValidationAccuracy;ValidationLoss;ValidationMAE;ValidationMAEStd;TestAccuracy;TestLoss;TestMAE;TestMAEStd\n"
        else:
            header = "Dataset;RunNumber;ValidationNumber;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss;EpochAccuracy;" \
                     "EpochTime;ValidationAccuracy;ValidationLoss;TestAccuracy;TestLoss\n"

        # Save file for results and add header if the file is new
        with open(f'{self.results_path}{self.para.db}/Results/{file_name}', "a") as file_obj:
            if os.stat(f'{self.results_path}{self.para.db}/Results/{file_name}').st_size == 0:
                file_obj.write(header)

        """
        Variable learning rate
        """
        scheduler_on = self.para.configs['scheduler']
        if scheduler_on:
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        """
        Store the best epoch
        """
        best_epoch = {"epoch": 0, "acc": 0.0, "loss": 100.0, "val_acc": 0.0, "val_loss": 100.0, "val_mae": 1000000}

        """
        Run through the defined number of epochs
        """
        for epoch in range(self.para.n_epochs):

            # Test stopping criterion
            if self.para.configs['early_stopping']['enabled']:
                if epoch - best_epoch["epoch"] > self.para.configs['early_stopping']['patience']:
                    if self.para.print_results:
                        print(f"Early stopping at epoch {epoch}")
                    break

            timer.measure("epoch")
            net.epoch = epoch
            epoch_loss = 0.0
            running_loss = 0.0
            epoch_acc = 0.0
            epoch_mae = 0.0
            epoch_mae_std = 0.0

            """
            Random Train batches for each epoch
            """
            np.random.seed(epoch + 687497)
            np.random.shuffle(self.training_data)
            train_batches = np.array_split(self.training_data, self.training_data.size // self.para.batch_size)

            for batch_counter, batch in enumerate(train_batches, 0):
                timer.measure("forward")
                optimizer.zero_grad()
                outputs = Variable(torch.zeros((len(batch), self.graph_data.num_classes), dtype=dtype))

                # TODO batch in one matrix ?
                for j, graph_id in enumerate(batch, 0):
                    net.train(True)
                    timer.measure("forward_step")
                    if 'random_variation' in self.para.configs and self.para.configs['random_variation']:
                        scale = 1.0
                        # random variation as torch tensor
                        random_variation = np.random.normal(0, scale, self.graph_data.inputs[graph_id].shape)
                        if 'precision' in self.para.configs and self.para.configs['precision'] == 'float':
                            random_variation = torch.FloatTensor(random_variation)
                        else:
                            random_variation = torch.DoubleTensor(random_variation)
                        outputs[j] = net(random_variation, graph_id)
                    else:
                        outputs[j] = net(self.graph_data.inputs[graph_id].to(device), graph_id)
                    timer.measure("forward_step")

                labels = self.graph_data.one_hot_labels[batch]

                loss = criterion(outputs, labels)
                timer.measure("forward")

                weights = []
                if self.para.save_weights:
                    for i, layer in enumerate(net.net_layers):
                        weights.append([x.item() for x in layer.Param_W])
                        w = np.array(weights[-1]).reshape(1, -1)
                        df = pd.DataFrame(w)
                        df.to_csv(f"Results/Parameter/layer_{i}_weights.csv", header=False, index=False, mode='a')

                timer.measure("backward")
                # change learning rate with high loss
                #for g in optimizer.param_groups:
                #    loss_value = loss.item()
                #    min_val = 50 - epoch ** (1. / 6.) * (49 / self.para.n_epochs ** (1. / 6.))
                #    loss_val = 100 * loss_value ** 2
                    #learning_rate_mul = min(min_val, loss_val)
                    #g['lr'] = self.para.learning_rate * learning_rate_mul
                    # print min_val, loss_val, learning_rate_mul, g['lr']
                #    if self.para.print_results:
                #        print(f'Min: {min_val}, Loss: {loss_val}, Learning rate: {g["lr"]}')

                loss.backward()
                optimizer.step()
                timer.measure("backward")
                timer.reset()

                if self.para.save_weights:
                    weight_changes = []
                    for i, layer in enumerate(net.net_layers):
                        change = np.array([weights[i][j] - x.item() for j, x in enumerate(layer.Param_W)]).flatten().reshape(1, -1)
                        weight_changes.append(change)
                        # save to three differen csv files using pandas
                        df = pd.DataFrame(change)
                        df.to_csv(f'Results/Parameter/layer_{i}_change.csv', header=False, index=False, mode='a')
                        # if there is some change print that the layer trains
                        if np.count_nonzero(change) > 0:
                            print(f'Layer {i} has updated')
                        else:
                            print(f'Layer {i} has not updated')

                running_loss += loss.item()
                epoch_loss += running_loss






                '''
                Evaluate the training accuracy
                '''
                batch_acc = 100 * ttd.get_accuracy(outputs, labels, one_hot_encoding=True)
                epoch_acc += batch_acc * (len(batch) / len(self.training_data))
                batch_mae = 0
                batch_mae_std = 0
                # if num classes is one calculate the mae and mae_std or if the task is regression
                if self.graph_data.num_classes == 1 or self.para.run_config.task == 'regression':
                    # flatten the labels and outputs
                    flatten_labels = labels.flatten().detach().numpy()
                    flatten_outputs = outputs.flatten().detach().numpy()
                    batch_mae = np.mean(np.abs(flatten_labels - flatten_outputs))
                    batch_mae_std = np.std(np.abs(flatten_labels - flatten_outputs))
                    epoch_mae += np.mean(np.abs(flatten_labels - flatten_outputs)) * (len(batch) / len(self.training_data))
                    epoch_mae_std += np.std(np.abs(flatten_labels - flatten_outputs)) * (len(batch) / len(self.training_data))

                if self.para.print_results:
                    if self.graph_data.num_classes == 1 or self.para.run_config.task == 'regression':
                        print("\tepoch: {}/{}, batch: {}/{}, loss: {}, acc: {} %, mae: {}, mae_std: {}".format(epoch + 1, self.para.n_epochs,
                                                                                      batch_counter + 1,
                                                                                      len(train_batches),running_loss, batch_acc, batch_mae, batch_mae_std))
                    else:
                        print("\tepoch: {}/{}, batch: {}/{}, loss: {}, acc: {} % ".format(epoch + 1, self.para.n_epochs,
                                                                                      batch_counter + 1,
                                                                                      len(train_batches),
                                                                                      running_loss, batch_acc))
                self.para.count += 1
                running_loss = 0.0

                if self.para.save_prediction_values:
                    # print outputs and labels to a csv file
                    outputs_np = outputs.detach().numpy()
                    # transpose the numpy array
                    outputs_np = outputs_np.T
                    df = pd.DataFrame(outputs_np)
                    # show only two decimal places
                    df = df.round(2)
                    df.to_csv("Results/Parameter/training_predictions.csv", header=False, index=False, mode='a')
                    labels_np = labels.detach().numpy()
                    labels_np = labels_np.T
                    df = pd.DataFrame(labels_np)
                    df.to_csv("Results/Parameter/training_predictions.csv", header=False, index=False, mode='a')

            # prune each five epochs
            if (epoch + 1) % 10 == 0 and 0 < epoch + 1 < self.para.n_epochs and 'prune' in self.para.configs and self.para.configs['prune']:
                print('Pruning')
                # iterate over the layers of the neural net
                for layer in net.net_layers:
                    if layer.name != 'Resize_Layer':
                        # get tensor from the parameter_list layer.Param_W
                        layer_tensor = torch.abs(torch.tensor(layer.Param_W))
                        # print number of non zero entries in layer_tensor
                        print(f'Number of non zero entries in layer {layer.name}: {torch.count_nonzero(layer_tensor)}')
                        # get the indices of the trainable parameters with lowest absolute max(1, 1%)
                        k = max(1, int(layer_tensor.size(0) * 0.20))
                        low = torch.topk(layer_tensor, k, largest=False)
                        lowest_indices = get_k_lowest_nonzero_indices(layer_tensor, k)
                        # print the number of lowest indices
                        print(f'Number of lowest indices in layer {layer.name}: {len(lowest_indices)}')
                        # iterate over the entries of the lowest indices
                        for i in lowest_indices:
                            # set the lowest indices to zero and torch.grad to false
                            layer.Param_W[i].requires_grad_(False)
                            layer.Param_W[i][:] = 0
                        # replace all non zero entries with the original values
                        layer_tensor = torch.abs(torch.tensor(layer.Param_W))
                        non_zero = torch.nonzero(layer_tensor, as_tuple=True)
                        # print the number of non zero entries
                        print(f'Number of non zero entries in layer {layer.name}: {len(non_zero[0])}')
                        for i in non_zero[0]:
                            with torch.no_grad():
                                layer.Param_W[i][:] = layer.Param_W_original[i]
                            # enable grad for the non zero entries
                            layer.Param_W[i].requires_grad_(True)
            '''
            Evaluate the validation accuracy for each epoch
            '''
            validation_acc = 0
            validation_mae = 0
            validation_mae_std = 0
            if self.validate_data.size != 0:
                outputs = torch.zeros((len(self.validate_data), self.graph_data.num_classes), dtype=dtype)
                # use torch no grad to save memory
                with torch.no_grad():
                    for j, data_pos in enumerate(self.validate_data):
                        net.train(False)
                        outputs[j] = net(self.graph_data.inputs[data_pos].to(device), data_pos)
                labels = self.graph_data.one_hot_labels[self.validate_data]
                # get validation loss
                validation_loss = criterion(outputs, labels).item()
                labels_argmax = labels.argmax(axis=1)
                outputs_argmax = outputs.argmax(axis=1)
                validation_acc = 100 * sklearn.metrics.accuracy_score(labels_argmax, outputs_argmax)
                if self.graph_data.num_classes == 1:
                    flatten_labels = labels.flatten().detach().numpy()
                    flatten_outputs = outputs.flatten().detach().numpy()
                    validation_mae = np.mean(np.abs(flatten_labels - flatten_outputs))
                    validation_mae_std = np.std(np.abs(flatten_labels - flatten_outputs))

                # update best epoch
                if self.graph_data.num_classes == 1:
                    if validation_mae <= best_epoch["val_mae"]:
                        best_epoch["epoch"] = epoch
                        best_epoch["acc"] = epoch_acc
                        best_epoch["loss"] = epoch_loss
                        best_epoch["val_acc"] = validation_acc
                        best_epoch["val_loss"] = validation_loss
                        best_epoch["val_mae"] = validation_mae
                        best_epoch["val_mae_std"] = validation_mae_std
                        # save the best model
                        if not os.path.exists(f'{self.results_path}{self.para.db}/Models/'):
                            os.makedirs(f'{self.results_path}{self.para.db}/Models/')
                        # Save the model if best model is used
                        if 'best_model' in self.para.configs and self.para.configs['best_model']:
                            torch.save(net.state_dict(),
                                       f'{self.results_path}{self.para.db}/Models/model_{self.para.config_id}_run_{self.run_id}_val_step_{self.k_val}.pt')

                else:
                    if validation_acc > best_epoch["val_acc"] or validation_acc == best_epoch["val_acc"] and validation_loss < best_epoch["val_loss"]:
                        best_epoch["epoch"] = epoch
                        best_epoch["acc"] = epoch_acc
                        best_epoch["loss"] = epoch_loss
                        best_epoch["val_acc"] = validation_acc
                        best_epoch["val_loss"] = validation_loss
                        # save the best model
                        if not os.path.exists(f'{self.results_path}{self.para.db}/Models/'):
                            os.makedirs(f'{self.results_path}{self.para.db}/Models/')
                        # Save the model if best model is used
                        if 'best_model' in self.para.configs and self.para.configs['best_model']:
                            torch.save(net.state_dict(),
                                       f'{self.results_path}{self.para.db}/Models/model_{self.para.config_id}_run_{self.run_id}_val_step_{self.k_val}.pt')

            if self.para.save_prediction_values:
                # print outputs and labels to a csv file
                outputs_np = outputs.detach().numpy()
                # transpose the numpy array
                outputs_np = outputs_np.T
                df = pd.DataFrame(outputs_np)
                # show only two decimal places
                df = df.round(2)
                df.to_csv("Results/Parameter/validation_predictions.csv", header=False, index=False, mode='a')
                labels_np = labels.detach().numpy()
                labels_np = labels_np.T
                df = pd.DataFrame(labels_np)
                df.to_csv("Results/Parameter/validation_predictions.csv", header=False, index=False, mode='a')

            # Test accuracy
            # print only if run best model is used
            test = False
            test_acc = 0
            test_loss = 0
            test_mae = 0
            test_mae_std = 0
            if 'best_model' in self.para.configs and self.para.configs['best_model']:
                # Test accuracy
                outputs = torch.zeros((len(self.test_data), self.graph_data.num_classes), dtype=dtype)
                with torch.no_grad():
                    for j, data_pos in enumerate(self.test_data, 0):
                        net.train(False)
                        outputs[j] = net(self.graph_data.inputs[data_pos].to(device), data_pos)
                labels = self.graph_data.one_hot_labels[self.test_data]
                test_loss = criterion(outputs, labels).item()
                test_acc = 100 * sklearn.metrics.accuracy_score(labels.argmax(axis=1), outputs.argmax(axis=1))
                test_mae = 0
                test_mae_std = 0
                if self.graph_data.num_classes == 1:
                    flatten_labels = labels.flatten().detach().numpy()
                    flatten_outputs = outputs.flatten().detach().numpy()
                    test_mae = np.mean(np.abs(flatten_labels - flatten_outputs))
                    test_mae_std = np.std(np.abs(flatten_labels - flatten_outputs))
                if self.para.print_results:
                    np_labels = labels.detach().numpy()
                    np_outputs = outputs.detach().numpy()
                    # np array of correct/incorrect predictions
                    labels_argmax = np_labels.argmax(axis=1)
                    outputs_argmax = np_outputs.argmax(axis=1)
                    np_correct = labels_argmax == outputs_argmax
                    # print entries of np_labels and np_outputs
                    for j, data_pos in enumerate(self.test_data, 0):
                        print(data_pos, np_labels[j], np_outputs[j], np_correct[j])

                if self.para.save_prediction_values:
                    # print outputs and labels to a csv file
                    outputs_np = outputs.detach().numpy()
                    # transpose the numpy array
                    outputs_np = outputs_np.T
                    df = pd.DataFrame(outputs_np)
                    # show only two decimal places
                    df = df.round(2)
                    df.to_csv("Results/Parameter/test_predictions.csv", header=False, index=False, mode='a')
                    labels_np = labels.detach().numpy()
                    labels_np = labels_np.T
                    df = pd.DataFrame(labels_np)
                    df.to_csv("Results/Parameter/test_predictions.csv", header=False, index=False, mode='a')

            timer.measure("epoch")
            epoch_time = timer.get_flag_time("epoch")
            if self.para.print_results:

                # if class num is one print the mae and mse
                if self.graph_data.num_classes == 1:
                    print(f'run: {self.run_id} val step: {self.k_val} epoch: {epoch + 1}/{self.para.n_epochs} epoch loss: {epoch_loss} epoch acc: {epoch_acc} epoch mae: {epoch_mae} +- {epoch_mae_std} epoch time: {epoch_time}'
                            f' validation acc: {validation_acc} validation loss: {validation_loss} validation mae: {validation_mae} +- {validation_mae_std}'
                          f'test acc: {test_acc} test loss: {test_loss} test mae: {test_mae} +- {test_mae_std}'
                          f'time: {epoch_time}')
                else:
                    print(f'run: {self.run_id} val step: {self.k_val} epoch: {epoch + 1}/{self.para.n_epochs} epoch loss: {epoch_loss} epoch acc: {epoch_acc}'
                            f' validation acc: {validation_acc} validation loss: {validation_loss}'
                            f'test acc: {test_acc} test loss: {test_loss}'
                            f'time: {epoch_time}')


            if self.graph_data.num_classes == 1:
                res_str = f"{self.para.db};{self.run_id};{self.k_val};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                          f"{epoch_loss};{epoch_acc};{epoch_mae};{epoch_mae_std};{epoch_time};{validation_acc};{validation_loss};{validation_mae};{validation_mae_std};{test_acc};{test_loss};{test_mae};{test_mae_std}\n"
            else:
                res_str = f"{self.para.db};{self.run_id};{self.k_val};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                          f"{epoch_loss};{epoch_acc};{epoch_time};{validation_acc};{validation_loss};{test_acc};{test_loss}\n"

            # Save file for results
            with open(f'{self.results_path}{self.para.db}/Results/{file_name}', "a") as file_obj:
                file_obj.write(res_str)

            if self.para.draw:
                self.para.draw_data = ttd.plot_learning_data(epoch + 1,
                                                             [epoch_acc, validation_acc, test_acc, epoch_loss],
                                                             self.para.draw_data, self.para.n_epochs)

            if scheduler_on:
                # check if learning rate is > 0.0001
                if optimizer.param_groups[0]['lr'] > 0.0001:
                    scheduler.step()


