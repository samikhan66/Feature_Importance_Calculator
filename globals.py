import configparser
from helpers import parser

options = configparser.ConfigParser()
options.read('options.ini')

dataset_option = parser.string2bool(options["data"]["dataset"])

drop_na = parser.string2bool(options["preprocess"]["drop_na"])

visualize_correlation = parser.string2bool(options["visualization"]["correlation_plot"])

feature_selection_method = parser.string2bool(options["model"]["feature_selection"])
ml_method = parser.string2bool(options["model"]["ml_method"])

action = parser.string2bool(options["model"]["action"])

csv_path = parser.string2bool(options["preprocess"]["csv_path"])
