# SEEDConsistencyChecker

## Getting Started

### Play with a certain method? (`CMIF` for example)

1. Clone the repository:
   ```
   git clone https://github.com/SEEDGuard/SEEDConsistencyChecker.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the file from `SEEDConsistencyChecker` directory:

   ```
   cd SEEDConsistencyChecker
   python main.py --method CMIF --input_dir 'path/to/input/folder/' --output_dir 'path/to/output/folder/'
   ```

   Output is stored in the `path/to/output/folder/` directory.

### To run the Testing script

Steps 1 and 2 will be the same.

3. Configure the input and output path in `testing.py`:

   ```
   input_dir = 'path/to/input/folder/'
   output_dir = 'path/to/output/folder/'
   ``` 

4. Run the testing script from `SEEDConsistencyChecker` directory:

   ```
   cd SEEDConsistencyChecker
   python test/CMIF/testing.py
   ```
### Using Docker

1. Clone the repository:
   ```
   git clone https://github.com/SEEDGuard/SEEDConsistencyChecker.git
   ```
2. Make sure you have Docker installed and started, follow if not [Install Docker](https://docs.docker.com/engine/install/).

3. Navigate to the folder consisting of `Dockerfile`

4. Build Docker Image
   ```
   docker build --tag your-image-name  .
   ```
5. Run the Docker image inside container  
   Update your-image-name, your_method, dataset according to your configuration. For more info on available methods refer to our [Methods](https://github.com/SEEDGuard/SEEDUtils/blob/main/README.md) list provided in README
   ```
   -docker run -it your-image-name --method your_method --input_dir dataset/input/ --output_dir dataset/output/
   ```
   Example:
   
   ```
   docker build -t cmif:1.1 .
   
   docker run -it cmif:1.1
   ```
6. Application is up and running. Output is generated and you can see them in the test folder 
7. You can see the logs, files, status for the container on the Docker Desktop.


## Contributing

SEEDConsistencyChecker thrives on community contributions. Whether you're interested in enhancing its security features, expanding the API, or improving the current functionality, your contributions are welcome. Please refer to our contribution guideline at [CONTRIBUTING.md](https://github.com/SEEDGuard/SEEDConsistencyChecker/blob/main/CONTRIBUTING.md) for more information on how to contribute. Also refer to our [Docker](https://github.com/SEEDGuard/SEEDUtils/blob/main/template/Dockerfile) template if you are coming up with new Methods for the task.

## Paper List

| Year-Id | Title                                                                                                                               | Venue Name | Replication Package                                                  | If Integrated  | If confirmed with the orignal author? | Current Contributors| 
| ------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------- | ------------- | --------- | --------- |
| 2024-2  | [Code Comment Inconsistency Detection Based on Confidence Learning](https://ieeexplore.ieee.org/abstract/document/10416264)         | TSE        | [Link](https://github.com/seekerstrive/MCCL)                         |               |               | [mrhuggins03](https://github.com/SEEDGuard/SEEDConsistencyChecker/tree/mitchell), David Wang
| 2024-1  | [How are We Detecting Inconsistent Method Names? An Empirical Study from Code Review Perspective](https://arxiv.org/abs/2308.12701) | arXiv      | [Link](https://figshare.com/s/8cdb4e3208e01991e45c)                  |               |               |
| 2023-4  | [Inconsistent Defect Labels: Essence, Causes, and Influence.](https://doi.org/10.1109/TSE.2022.3156787)                             | TSE        |
| 2023-3  | [Keeping Mutation Test Suites Consistent and Relevant with Long-Standing Mutants.](https://doi.org/10.1145/3611643.3613089)         | FSE        |
| 2023-2  | [Data Quality Matters: A Case Study of Obsolete Comment Detection](https://ieeexplore.ieee.org/abstract/document/10172689)          | ICSE       | [Link](https://github.com/SoftWiser-group/AdvOC)                     |               |               | Kriti Patnala
| 2023-1  | [When to Say What: Learning to Find Condition-Message Inconsistencies](https://ieeexplore.ieee.org/abstract/document/10172811)      | ICSE       | [Link](https://zenodo.org/records/7624781)                           | :heavy_check_mark: |               | [EZ7051](https://github.com/SEEDGuard/SEEDConsistencyChecker/tree/main/core/CMIF)
|2021-2 |[A Context-based Automated Approach for Method Name Consistency Checking and Suggestion.](https://doi.org/10.1109/ICSE43902.2021.00060)  |ICSE   |
| 2021-1  | [Deep Just-In-Time Inconsistency Detection Between Comments and Source Code](https://arxiv.org/pdf/2010.01625.pdf)                  | AAAI       | [Link](https://github.com/panthap2/deep-jit-inconsistency-detection) |               |               | [vigneskv](https://github.com/SEEDGuard/SEEDConsistencyChecker/tree/vigneskv)
| 2020-1  | [A Fine-Grained Analysis on the Inconsistent Changes in Code Clones.](https://doi.org/10.1109/ICSME46990.2020.00030)                | ICSME      |
| 2019-1  | [A Large-Scale Empirical Study on Code-Comment Inconsistencies](https://doi.org/10.1109/ICPC.2019.00019)                            | ICPC       |                                                                      |               |
