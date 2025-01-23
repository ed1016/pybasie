# PyBasie - Psychometric function evaluation using Basie in Python
PyBasie is a Python-based GUI tool for evaluating Speech Reception Thresholds (SRTs) in (reverberant) speech-in-noise conditions. It leverages a Bayesian adaptive method to perform precise psychometric function evaluations, following the methodology outlined in [1]. This tool is designed for researchers and practitioners in auditory and psychophysical studies and was used in the study presented in [to come]

<p float="none" align="middle">
  <img src="docs/gui.png" width="50%" hspace="2%"/>
</p>

> **Figure 1**: Preview of the GUI used for SRT estimation.


## Requirements
- **Python** (get it [here](https://www.python.org/downloads)).
- **Github** (get it [here](https://git-scm.com/download/win)).
- **[Optional] MATLAB** to reproduce the experiments in [to come].

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ed1016/pybasie.git
   ```
2. Install dependencies: 
   ```bash
   cd pybasie
   pip install -r requirements.txt
   ```
3. Launch the GUI:
   ```bash
   python psycest_gui.py
   ```
## Usage
This repository contains a minimum working example in the `exampledata` folder. 
   
## Detailed Features
<p float="none" align="middle">
  <img src="docs/guiannotation.png" width="100%" hspace="2%"/>
</p>
> **Figure 2**: GUI used for SRT estimation where (A) shows the experiment parameters panel, (B) indicates where the Basie parameters for SRT estimation can be adjusted, (C) are the start/stop buttons, and (D) is the control panel showing progress with trials.  

The GUI contains 4 main panels, labelled in Figure 2:
- The **Parameters** panel **A** is used to set the experiment parameters where
  - **Subject ID** is set to link results to a participant's name.
  - **Audio files** is the path to the _folder_ containing the audio to use in the experiment.
  - **Ouput directory** is the path to the _folder_ used to save results.
  - **Test type** is the type of test and feedback used. The current implementation contains the MRT sentences [2,3] with the option to type the participant's answer, or select an option out of five.
  - **Sentence mapping** is the path to the _file_ containing the audio file-to-word mapping used in the experiment (e.g. `mrt_001.wav` is the word `went`).
- The **Advanced parameters** panel **B** contains the parameters specific to Basie [1] for SRT estimation:
  - **Slope weight** determines how much importance the algorithm should put on the slope estimation compared to the SRT estimation.
  - **Min SNR (dB)** is the minimum available SNR in the audio, expressed in dB.
  - **Max SNR (dB)** is the maximum available SNR in the audio, expressed in dB.
  - **Guess rate** is the assumed rate at which subjects can correctly guess the word without hearing it.
  - **Miss rate** is the assumed rate at which subjects incorrectly repeat the word even if they have heard it.
  - **Max. nbr. trials** is the number of trials after which the experiment stops.
- The bottom panel **C** contains buttons used to start the experiment:
  - **Calibrate** gives the option to playback noiseless audio and the loudest sample in the experiment to ensure the sound is set at an intelligible and comfortable level.
  - **Practice** gives particpants the opportunity to familiarise themselves with the experiments.
  - **Start** begins the experiment.
- The right-side panel **D** showcases information as the experiment progresses:
  - In each trial, the name of the audio file being played is printed at the top of the panel.
  - Each probe SNR is progressively plotted using a dashed line and crosses. The value of the SRT estimate at each step is plotted as an horizontal line.
  - The exact value of the SRT and slope estimates is given at the bottom of the panel, along with the associated estimate variance. 



## Experiment reproduction [Optional]
Reproduce the experimental setup in [to come] using the MATLAB scripts provided in the `utils` folder:
- `create_babble.m`: Generates the babble noise.
- `generate_data.m`: Creates the reverberant speech-in-noise files.
- `sentence_mapping.txt`: Contains the mapping from audio file to spoken sentence.

## Detailed Features

## To do
- [ ] Give details on GUI usage
- [ ] Run MC simulations (and provide code) to show convergence VS matlab and VS usual methods
- [ ] Write up latex doc with more details about the method 


 ## Citing
 Please cite this work using
 ```
@article{Doire2017,
  title={Robust and efficient Bayesian adaptive psychometric function estimation},
  author={Doire, Clement SJ and Brookes, Mike and Naylor, Patrick A},
  journal={The Journal of the Acoustical Society of America},
  volume={141},
  number={4},
  pages={2501--2512},
  year={2017}
}
```


## References
[1]&nbsp; Doire, C.S., Brookes, M. and Naylor, P.A., 2017. [Robust and efficient Bayesian adaptive psychometric function estimation](https://pubs.aip.org/asa/jasa/article/141/4/2501/1059157). The Journal of the Acoustical Society of America, 141(4), pp.2501-2512.<br>
[2]&nbsp; House, A.S., Williams, C., Hecker, M.H. and Kryter, K.D., 1963. [Psychoacoustic speech tests: A modified rhyme test](https://pubs.aip.org/asa/jasa/article/35/11_Supplement/1899/617588). The Journal of the Acoustical Society of America, 35(11_Supplement), pp.1899-1899.<br>
[3]&nbsp; Cooke, M., Valentini-Botinhao, C. and Mayo, C., 2019. [Hurricane natural speech corpus-higher quality version](https://datashare.ed.ac.uk/handle/10283/347).
 
