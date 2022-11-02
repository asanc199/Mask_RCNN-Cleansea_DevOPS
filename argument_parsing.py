import argparse

from traitlets import default



PRETRAIN_OPTIONS = ['coco', 'imagenet', 'last']
AUGMENTATION_OPTIONS = ['none', 'mild', 'severe']
TRAIN_OPTIONS = ['real', 'synth']
PROCESS_OPTIONS = ['train', 'inference', 'both']
FILLING_OPTIONS = ['none', 'synth']


"""Arguments menu"""
def menu():
	parser = argparse.ArgumentParser(description='CleanSea experiments')

	parser.add_argument('-pretrain',    dest="pretrain",                required = False,			help='Pretrain corpus', choices = PRETRAIN_OPTIONS, default = PRETRAIN_OPTIONS[0])
	parser.add_argument('-train_db',	dest="train_db",                required = True,			help='Train data', choices = TRAIN_OPTIONS)
	parser.add_argument('-test_db',     dest="test_db",					required = True,            help='Test data', choices = TRAIN_OPTIONS)
	parser.add_argument('-process',		dest="process",					required = True,			help='Process to carry out', choices = PROCESS_OPTIONS)
	parser.add_argument('-fill_db',		dest="fill_db",					required = True,			help='Filling data', choices = FILLING_OPTIONS)
	parser.add_argument('-aug',         dest="augmentation",            required = True,			help='Augmentation type', choices = AUGMENTATION_OPTIONS)
	parser.add_argument('-size',        dest="size_perc",               required = False,			help='Train size percentage', default = 100, type = int)
	parser.add_argument('-fill_size',   dest="fill_size_perc",          required = False,			help='Fill size percentage (only if -limit_train=false)', default = 100, type = int)
	parser.add_argument('-epochs',		dest='epochs',					required = True,			help='List for the epoch breaks', type=str, default = '50, 100')
	parser.add_argument('-limit_train',	dest='limit_train',				required = False,			help='Limit amount of train data', type=str_to_bool, nargs='?', const=True, default=True)

	args = parser.parse_args()
	args.epochs = [int(item) for item in args.epochs.split(',')]
	args.fill_size_perc = args.fill_size_perc if args.limit_train == False else 100

	return args


def str_to_bool(value):
	if isinstance(value, bool):
		return value
	if value.lower() in {'false', 'f', '0', 'no', 'n'}:
		return False
	elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
		return True
	raise ValueError(f'{value} is not a valid boolean value')