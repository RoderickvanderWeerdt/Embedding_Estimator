import pandas as pd
import numpy as np

from dataset_KGbench import Emb_KGbench_Dataset, ToTensor
from torch.utils.data import DataLoader
import torch

import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def test(dataloader, model, device):
    num_batches = len(dataloader)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for sample in dataloader:
            X = sample['embedding']
            y = sample['target_class']
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            _, predicted = torch.max(pred.data, 1)
            for p_i, y_i in zip(predicted, y):
                print(int(p_i), int(y_i))
            total += y.size(0)
            correct += (predicted == y).sum().item()
    correct /= total
    return correct

def clean_emb_list(list_):
    list_ = list_[1:-1]
    list_ = np.fromstring(list_, dtype=float, sep=', ')
    return list_

def create_value_df(entities_df, skip_columns):
	value_df = pd.DataFrame()
	for col in entities_df.columns:
		if col in skip_columns: #skip none numeric columns
			continue
		value_df[col] = pd.to_numeric(entities_df[col], errors='coerce') #coerce makes nan out of none numeric values :D
	return value_df

def euclidean_for_df(numeric_df, new_data_df):
	error_df = (numeric_df - new_data_df).pow(2)
	error_sqrd_df = error_df.mean(axis=1).pow(1./2)
	return error_sqrd_df

def select_k_similar_embeddings(sorted_df, k, emb_column_name):
	selected_embeddings_array = sorted_df[emb_column_name][:k]
	return selected_embeddings_array

def most_frequent(List):
    return max(set(List), key = List.count)

def select_k_similar_classes(sorted_df, k, target_class):
	classes_of_selected_array = sorted_df['y'][:k]
	classes_of_selected_array = list(classes_of_selected_array)

	print(most_frequent(classes_of_selected_array))

def create_new_embedding(selected_embeddings_array):
	return selected_embeddings_array.mean(axis=0)

def create_average_embedding(df, emb_column_name):
	return df[emb_column_name].mean(axis=0)

def shuffle_dataset(dataset_fn):
    df = pd.read_csv(dataset_fn, sep=",")
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("dataset_fn", index=False)

def make_val_20(entities_fn, new_dp_fn):
	shuffle_dataset(new_dp_fn)
	entities_df = pd.read_csv(entities_fn)
	new_dp_df = pd.read_csv(new_dp_fn)[:int(len(entities_df)*0.2)]
	new_dp_df.to_csv(new_dp_fn, sep=",", index=False)


if __name__ == '__main__':
	am = False
	dmg= True
	md = False
	testing=False
	if am:
		# entities_fn = "data/amplus-stripped_without_edm_object_w_emb.csv"
		# new_dp_fn = "data/AM_entities_units_VALIDATION200.csv"
		# new_dp_fn = "data/AM_entities_units_VALIDATION.csv"
		# model_fn = "data/amplus-stripped_without_edm_object_w_emb.pytorch-model"
		entities_fn = "data/AM_entities_units_w_emb.csv"
		model_fn = "data/AM_entities_units_w_emb.pytorch-model"
		new_dp_fn = "data/AM_entities_units_VALIDATION_w_emb_smaller.csv"
	elif dmg:
		# entities_fn = "data/dmg777k_stripped_without_thumbs_geo_w_emb.csv"
		# new_dp_fn = "data/entities_dgm777k_VALIDATION.csv"
		# model_fn = "data/dmg777k_stripped_without_thumbs_geo_w_emb.pytorch-model"
		entities_fn = "data/entities_dmg777k_TRAIN_w_emb.csv"
		new_dp_fn = "data/entities_dmg777k_VALIDATION_w_emb.csv"
		model_fn = "data/entities_dmg777k_TRAIN_w_emb.pytorch-model"
	elif md:
		entities_fn = "data/md_raw_without_Images_w_emb.csv"
		# new_dp_fn = "data/entities_md_raw_VALIDATION.csv"
		# model_fn = "data/md_raw_without_Images_w_emb.pytorch-model"
		entities_fn = "data/entities_md_raw_TRAIN_w_emb.csv"
		# new_dp_fn = "data/entities_md_raw_VALIDATION_w_emb.csv"
		new_dp_fn = "data/entities_md_raw_VALIDATION_w_emb.csv"
		model_fn = "data/md_raw_without_Images_w_emb.pytorch-model"
	elif testing:
		entities_fn = "data/AM_entities_units_w_emb_SMALL.csv"
		new_dp_fn = "data/AM_entities_units_VALIDATION_w_emb_SMALL.csv"



	k_similar = 10
	entity_column_name = "entity"
	class_column_name = "y"
	emb_column_name = "emb"
	update_emb_column_name = "update_emb"
	skip_columns = [entity_column_name, class_column_name, emb_column_name, update_emb_column_name]
	create_new_embeddings = True
	use_classifier = True
	make_validation_20percent = False

	if make_validation_20percent:
		make_val_20(entities_fn, new_dp_fn)

	if create_new_embeddings:
		entities_df = pd.read_csv(entities_fn)
		entities_df[emb_column_name] = entities_df[emb_column_name].apply(clean_emb_list)

		value_df = create_value_df(entities_df, skip_columns)

		new_dp_df = pd.read_csv(new_dp_fn)
		new_dp_value_df = create_value_df(new_dp_df, skip_columns)

		# average_emb = create_average_embedding(entities_df, emb_column_name).tolist() #in case no better similar nodes are found?

		new_embeddings = []
		non_similar_counts = 0
		for i, new_datapoint in new_dp_value_df.iterrows():
			mae_df = euclidean_for_df(value_df, new_datapoint)
			if np.isnan(mae_df.mean()):
				# print("couldn't find any similar data points.")
				new_dp_df = new_dp_df.drop(i) #when a datapoint without attributes is encountered we remove it from the validation set.
				non_similar_counts += 1
				# new_embeddings.append(average_emb)
				continue
			entities_df["similarity"] = mae_df
			sorted_df = entities_df.sort_values("similarity")
			similar_embeddings = select_k_similar_embeddings(sorted_df, k_similar, emb_column_name)
			# select_k_similar_classes(sorted_df, k_similar, class_column_name)
			new_embedding = create_new_embedding(similar_embeddings)
			new_embeddings.append(new_embedding.tolist())

		print("couldn't find similar datapoints for", non_similar_counts, "out of", len(new_dp_value_df))
		new_dp_df[update_emb_column_name] = new_embeddings
		new_dp_df.to_csv(new_dp_fn, sep=",", index=False)

	if use_classifier:
		batch_size = 1
		data = Emb_KGBench_Dataset(csv_file=new_dp_fn, transform=ToTensor(), train_all=True, emb_header=update_emb_column_name)
		dataloader = DataLoader(data, batch_size=batch_size)

		model = NeuralNetwork(data.get_embedding_size(), data.get_n_classes())
		# model = NeuralNetwork(data.get_embedding_size(), 8) #for AM, if not all 8 classes are in the random validation set
		model.load_state_dict(torch.load(model_fn))
		device = "cuda" if torch.cuda.is_available() else "cpu"

		accuracy = test(dataloader, model, device)
		print(f"Test Accuracy: {(100*accuracy):>0.2f}%")