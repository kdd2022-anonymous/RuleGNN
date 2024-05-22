# Preparation:
1. Clone the repository
2. Install the required packages using the environment.yml file
3. Download the TUDataset from here https://chrsmrrs.github.io/datasets/docs/datasets/ and place the extracted folder in GraphData/TUGraphs/
4. Preprocess the distances by running GraphData/Distances/save_distances.py

# Running the experiments
Using the ModelSelection.sh script, you can run the experiments for the different datasets and models. 
The DATABASE_NAMES and the CONFIG_FILE variable in the script can be adjusted to run the experiments for different datasets and models.

Possible combinations of DATABASE_NAMES and CONFIG_FILE are:

DATABASE_NAMES = ["NCI1", "NCI109", "Mutagenicity"] <br>
CONFIG_FILE = "Configs/config_NCI1.yml"

DATABASE_NAMES = ["DHFR"] <br>
CONFIG_FILE = "Configs/config_DHFR.yml"

DATABASE_NAMES = ["IMDB-BINARY", "IMDB-MULTI"] <br>
CONFIG_FILE = "Configs/config_IMDB-BINARY.yml"

DATABASE_NAMES = ["LongRings100"] <br>
CONFIG_FILE = "Configs/config_LongRings.yml"

DATABASE_NAMES = ["EvenOddRings2_16] <br>
CONFIG_FILE = "Configs/config_EvenOddRings.yml"

DATABASE_NAMES = ["EvenOddRingsCount16"] <br>
CONFIG_FILE = "Configs/config_EvenOddRingsCount.yml"

DATABASE_NAMES = ["Snowflakes"] <br>
CONFIG_FILE = "Configs/config_Snowflakes.yml"

Using the RunBestModels.sh script, the model with the best validation accuracy is chosen and the model is trained three times with different seeds.

# Evaluation
The results of the experiments can be evaluated running EvaluationFinal.py