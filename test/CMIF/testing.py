from core.CMIF.cmiFinder import CmiFinder


def main():
    input_dir = "/Users/ejazahmed/Documents/SEM2/IS/SEEDConsistencyChecker/test/CMIF/data/input"
    output_dir = "/Users/ejazahmed/Documents/SEM2/IS/SEEDConsistencyChecker/test/CMIF/data/output"

    checker: CmiFinder = CmiFinder()
    checker.consistency_checker(data_dir=input_dir, dest_dir=output_dir)

if __name__ == "__main__":
    main()