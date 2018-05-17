from optparse import OptionParser
print(type(None))
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train_file", help="CONLL train data file")
    parser.add_option("--test", dest="test_file", help="CONLL test data file")

    (options, args) = parser.parse_args()
    print(options)
    print(args)
    print(options.train_file)
    print('printing')