import argparse

from traitlets import default



PRETRAIN_OPTIONS = ['coco', 'imagenet', 'last']
AUGMENTATION_OPTIONS = ['none', 'mild', 'severe']
TRAIN_OPTIONS = ['real', 'synth']
PROCESS_OPTIONS = ['train', 'inference', 'both']

"""Arguments menu"""
def menu():
	parser = argparse.ArgumentParser(description='CleanSea experiments')

	parser.add_argument('-pretrain',    dest="pretrain",                required = False,			help='Pretrain corpus', choices = PRETRAIN_OPTIONS, default = PRETRAIN_OPTIONS[0])
	parser.add_argument('-train_db',	dest="train_db",                required = True,			help='Train data', choices = TRAIN_OPTIONS)
	parser.add_argument('-test_db',     dest="test_db",					required = True,            help='Test data', choices = TRAIN_OPTIONS)
	parser.add_argument('-process',		dest="process",					required = True,			help='Process to carry out', choices = PROCESS_OPTIONS)
	parser.add_argument('-fill_db',		dest="fill_db",					required = False,			help='Filling data', type=str)
	parser.add_argument('-aug',         dest="augmentation",            required = True,			help='Augmentation type', choices = AUGMENTATION_OPTIONS)
	parser.add_argument('-size',        dest="size_perc",               required = False,			help='Train size percentage', default = 100, type = int)
	parser.add_argument('-epochs',		dest='epochs',					required = True,			help='List for the epoch breaks', type=str, default = '50, 100')


	args = parser.parse_args()
	args.epochs = [int(item) for item in args.epochs.split(',')]

	return args

False