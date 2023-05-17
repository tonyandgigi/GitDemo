clear all
close all
clc

% Define the communication rate of this system as  [bits/channel use], where (n,k) means that the system sends one of  messages using n channel uses. The channel impairs encoded (i.e. transmitted) symbols to generate . The decoder (i.e. receiver) produces an estimate, , of the transmitted message, . 
% The input message is defined as a one-hot vector , which is defined as a vector whose elements are all zeros except the  one. The channel is additive white Gaussian noise (AWGN) that adds noise to achieve a given energy per bit to noise power density ratio, . 
% Define a (7,4) autoencoder network with energy normalization and a training  of 3 dB. In [1], authors showed that two fully connected layers for both the encoder (transmitter) and the decoder (receiver) provides the best results with minimal complexity. Input layer (featureInputLayer) accepts a one-hot vector of length M. The encoder has two fully connected layers (fullyConnectedLayer). The first one has M inputs and M outputs and is followed by an ReLU layer (reluLayer). The second fully connected layer has M inputs and n outputs and is followed by the normalization layer (helperAEWNormalizationLayer.m). The encoder layers are followed by the AWGN channel layer (helperAEWAWGNLayer.m). The output of the channel is passed to the decoder layers. The first decoder layer is a fully connected layer that has n inputs and M outputs and is followed by an ReLU layer. Second fully connected layer has M inputs and M outputs and is followed by a softmax layer (softmaxLayer), which outputs the probability of each M symbols. The classification layer (classificationLayer) outputs the most probable transmitted symbol from 0 to M-1. 
k = 4;    % number of input bits
M = 2^k;  % number of possible input symbols
n = 7;    % number of channel uses
SNR_dB = 3; % Eb/No in dB

wirelessAutoencoder = [
  featureInputLayer(M,"Name","One-hot input","Normalization","none")
  
  fullyConnectedLayer(M,"Name","fc_1")
  reluLayer("Name","relu_1")
  
  fullyConnectedLayer(n,"Name","fc_2")
  
  helperAEWNormalizationLayer("Method", "Energy", "Name", "wnorm")
  
  helperAEWAWGNLayer("Name","channel",...
    "NoiseMethod","EbNo",...
    "EbNo",SNR_dB,...
    "BitsPerSymbol",2,...
    "SignalPower",1)
  
  fullyConnectedLayer(M,"Name","fc_3")
  reluLayer("Name","relu_2")
  
  fullyConnectedLayer(M,"Name","fc_4")
  softmaxLayer("Name","softmax")
  
  classificationLayer("Name","classoutput")]
% The helperAEWTrainWirelessAutoencoder.m function defines such a network based on the (n,k), normalization method and the  values. The Wireless Autoencoder Training Function section shows the contents of the helperAEWTrainWirelessAutoencoder.m function. 

% Train Autoencoder
% Run the helperAEWTrainWirelessAutoencoder.m function to train a (2,2) autoencoder with energy normalization. This function uses the trainingOptions function to select 
% Adam (adaptive moment estimation) optimizer, 
% Initial learning rate of 0.01, 
% Maximum epochs of 15,
% Minibatch size of 20*M,
% Piecewise learning schedule with drop period of 10 and drop factor of 0.1.
% Then, the helperAEWTrainWirelessAutoencoder.m function runs the trainNetwork function to train the autoencoder network with the selected options. Finally, this function separates the network into encoder and decoder parts. Encoder starts with the input layer and ends after the normalization layer. Decoder starts after the channel layer and ends with the classification layer. A feature input layer is added at the beginning of the decoder. 
% Train the autoencoder with an  value that is low enough to result in some errors but not too low such that the training algorithm cannot extract any useful information from the received symbols, y. Set  to 3 dB. 
% Training an autoencoder may take several minutes. Set trainNow to false to use saved networks.
trainNow = false; %#ok<*NASGU>

n = 2;                      % number of channel uses
k = 2;                      % bits per data symbol
SNR_dB = 3;                   % dB
normalization = "Energy";   % Normalization "Energy" | "Average power"

if trainNow
  [txNet22e,rxNet22e,info22e,wirelessAutoEncoder22e] = ...
    helperAEWTrainWirelessAutoencoder(n,k,normalization,SNR_dB); %#ok<*UNRCH>
else
  load trainedNet_n2_k2_energy txNet rxNet info trainedNet
  txNet22e = txNet;
  rxNet22e = rxNet;
  info22e = info;
  wirelessAutoEncoder22e = trainedNet;
end

% Plot the traning progress. The validation accuracy quickly reaches more than 90% while the validation loss keeps slowly decreasing. This behavior shows that the training  value was low enough to cause some errors but not too low to avoid convergence. For definitions of validation accuracy and validation loss, see Monitor Deep Learning Training Progress section.
figure
helperAEWPlotTrainingPerformance(info22e)

% Use the plot object function of the trained network objects to show the layer graphs of the full autoencoder, the encoder network, i.e. the transmitter, and the decoder network, i.e. the receiver.  
figure
tiledlayout(2,2)
nexttile([2 1])
plot(wirelessAutoEncoder22e)
title('Autoencoder')
nexttile
plot(txNet22e)
title('Encoder/Tx')
nexttile
plot(rxNet22e)
title('Decoder/Rx')

% Plot Transmitted and Received Constellation
% Plot the constellation learned by the autoencoder to send symbols through the AWGN channel together with the received constellation. For a (2,2) configuration, autoencoder learns a QPSK () constellation with a phase rotation. The received constellation is basically the activation values at the output of the channel layer obtained using the activations function and treated as interleaved complex numbers. 
subplot(1,2,1)
helperAEWPlotConstellation(txNet22e)
title('Learned Constellation')
subplot(1,2,2)
helperAEWPlotReceivedConstellation(wirelessAutoEncoder22e)
title('Received Constellation')

% Simulate BLER Performance
% Simulate the block error rate (BLER) performance of the (2,2) autoencoder. Setup simulation parameters.
simParams.EbNoVec = 0:0.5:8;
simParams.MinNumErrors = 10;
simParams.MaxNumFrames = 300;
simParams.NumSymbolsPerFrame = 10000;
simParams.SignalPower = 1;
% Generate random integers in the [0 -1] range that represents  random information bits. Encode these information bits into complex symbols with helperAEWEncode.m function. The helperAEWEncode function runs the encoder part of the autoencoder then maps the real valued  vector into a complex valued  vector such that the odd and even elements are mapped into the in-phase and the quadrature component of a complex symbol, respectively, where . In other words, treat the  array as an interleaved complex array. 
% Pass the complex symbols through an AWGN channel. Decode the channel impaired complex symbols with the helperAEWDecode.m function. The following code runs the simulation for each  point for at least 10 block errors. To obtain more accurate results, increase minimum number of errors to at least 100. If Parallel Computing Toolbox™ is installed and a license is available, the simulation will run on a parallel pool. Compare the results with that of an uncoded QPSK system with block length 2.
EbNoVec = simParams.EbNoVec;
R = k/n;

M = 2^k;
BLER = zeros(size(EbNoVec));
parfor EbNoIdx = 1:length(EbNoVec)
  SNR_dB = EbNoVec(EbNoIdx) + 10*log10(R);
  chan = comm.AWGNChannel("BitsPerSymbol",2, ...
    "EbNo", SNR_dB, "SamplesPerSymbol", 1, "SignalPower", 1);

  numBlockErrors = 0;
  frameCnt = 0;
  while (numBlockErrors < simParams.MinNumErrors) ...
      && (frameCnt < simParams.MaxNumFrames) %#ok<PFBNS>

    d = randi([0 M-1],simParams.NumSymbolsPerFrame,1);    % Random information bits
    x = helperAEWEncode(d,txNet22e);                      % Encoder
    y = chan(x);                                          % Channel
    dHat = helperAEWDecode(y,rxNet22e);                   % Decoder

    numBlockErrors = numBlockErrors + sum(d ~= dHat);
    frameCnt = frameCnt + 1;
  end
  BLER(EbNoIdx) = numBlockErrors / (frameCnt*simParams.NumSymbolsPerFrame);
end
figure
semilogy(simParams.EbNoVec,BLER,'-')
hold on
qpsk22BLER = 1-(1-berawgn(simParams.EbNoVec,'psk',4,'nondiff')).^2;
semilogy(simParams.EbNoVec,qpsk22BLER,'--')
hold off
ylim([1e-4 1])
grid on
xlabel('E_b/N_o (dB)')
ylabel('BLER')
legend('AE (2,2)','QPSK (2,2)')
% The well formed constellation together with the BLER results show that training for 15 epochs is enough to get a satisfactory convergence. 

% Compare Constellation Diagrams
% Compare learned constellations of several autoencoders normalized to unit energy and unit average power. Train (2,4) autoencoder normalized to unit energy.
n = 2;      % number of channel uses
k = 4;      % bits per data symbol
SNR_dB = 3;   % dB
normalization = "Energy";
if trainNow
  [txNet24e,rxNet24e,info24e,wirelessAutoEncoder24e] = ...
    helperAEWTrainWirelessAutoencoder(n,k,normalization,SNR_dB);
else
  load trainedNet_n2_k4_energy txNet rxNet info trainedNet
  txNet24e = txNet;
  rxNet24e = rxNet;
  info24e = info;
  wirelessAutoEncoder24e = trainedNet;
end

% Train (2,4) autoencoder normalized to unit average power.
n = 2;      % number of channel uses
k = 4;      % bits per data symbol
SNR_dB = 3;   % dB
normalization = "Average power";
if trainNow
  [txNet24p,rxNet24p,info24p,wirelessAutoEncoder24p] = ...
    helperAEWTrainWirelessAutoencoder(n,k,normalization,SNR_dB);
else
  load trainedNet_n2_k4_power txNet rxNet info trainedNet
  txNet24p = txNet;
  rxNet24p = rxNet;
  info24p = info;
  wirelessAutoEncoder24p = trainedNet;
end

% Train (7,4) autoencoder normalized to unit energy.
n = 7;      % number of channel uses
k = 4;      % bits per data symbol
SNR_dB = 3;   % dB
normalization = "Energy";
if trainNow
  [txNet74e,rxNet74e,info74e,wirelessAutoEncoder74e] = ...
    helperAEWTrainWirelessAutoencoder(n,k,normalization,SNR_dB);
else
  load trainedNet_n7_k4_energy txNet rxNet info trainedNet
  txNet74e = txNet;
  rxNet74e = rxNet;
  info74e = info;
  wirelessAutoEncoder74e = trainedNet;
end

% Plot the constellation using the helperAEWPlotConstellation.m function. The trained (2,2) autoencoder converges on a QPSK constellation with a phase shift as the optimal constellation for the channel conditions experienced. The (2,4) autoencoder with energy normalization converges to a 16PSK constellation with a phase shift. Note that, energy normalization forces every symbol to have unit energy and places the symbols on the unit circle. Given this constraint, best constellation is a PSK constellation with equal angular distance between symbols. The (2,4) autoencoder with average power normalization converges to a three-tier constellation of 1-6-9 symbols. Average power normalization forces the symbols to have unity average power over time. This constraint results in an APSK constellation, which is different than the conventional QAM or APSK schemes. Note that, this network configuration may also converge to a two-tier constellation with 7-9 symbols based on the random initial condition used during training. The last plot shows the 2-D mapping of the 7-D constellation generated by the (7,4) autoencoder with energy constraint. 2-D mapping is obtained using the t-Distributed Stochastic Neighbor Embedding (t-SNE) method (see tsne function).  
figure
subplot(2,2,1)
helperAEWPlotConstellation(txNet22e)
title('(2,2) Energy')
subplot(2,2,2)
helperAEWPlotConstellation(txNet24e)
title('(2,4) Energy')
subplot(2,2,3)
helperAEWPlotConstellation(txNet24p)
title('(2,4) Average Power')
subplot(2,2,4)
helperAEWPlotConstellation(txNet74e,'t-sne')
title('(7,4) Energy')

% Compare BLER Performance of Autoencoders with Coded and Uncoded QPSK
% Simulate the BLER performance of a (7,4) autoencoder with that of (7,4) Hamming code with QPSK modulation for both hard decision and maximum likelihood (ML) decoding. Use uncoded (4,4) QPSK as a baseline. (4,4) uncoded QPSK is basically a QPSK modulated system that sends blocks of 4 bits and measures BLER. The data for the following figures is obtained using helperAEWSimulateBLER.mlx and helperAEWPrepareAutoencoders.mlx files. 
load codedBLERResults.mat
figure
qpsk44BLERTh = 1-(1-berawgn(simParams.EbNoVec,'psk',4,'nondiff')).^4;
semilogy(simParams.EbNoVec,qpsk44BLERTh,':*')
hold on
semilogy(simParams.EbNoVec,qpsk44BLER,':o')
semilogy(simParams.EbNoVec,hammingHard74BLER,'--s')
semilogy(simParams.EbNoVec,ae74eBLER,'-')
semilogy(simParams.EbNoVec,hammingML74BLER,'--d')
hold off
ylim([1e-5 1])
grid on
xlabel('E_b/N_o (dB)')
ylabel('BLER')
legend('Theoretical Uncoded QPSK (4,4)','Uncoded QPSK (4,4)','Hamming (7,4) Hard Decision',...
  'Autoencoder (7,4)','Hamming (7,4) ML','Location','southwest')
title('BLER comparison of (7,4) Autoencoder')
% As expected, hard decision (7,4) Hamming code with QPSK modulation provides about 0.6 dB  advantage over uncoded QPSK, while the ML decoding of (7,4) Hamming code with QPSK modulation provides another 1.5 dB advantage for a BLER of . The (7,4) autoencoder BLER performance approaches the ML decoding of (7,4) Hamming code, when trained with 3 dB . This BLER performance shows that the autoencoder is able to learn not only modulation but also channel coding to achieve a coding gain of about 2 dB for a coding rate of R=4/7. 
% Next, simulate the BLER performance of autoencoders with R=1 with that of uncoded QPSK systems. Use uncoded (2,2) and (8,8) QPSK as baselines. Compare BLER performance of these systems with that of (2,2), (4,4) and (8,8) autoencoders. 
load uncodedBLERResults.mat
qpsk22BLERTh = 1-(1-berawgn(simParams.EbNoVec,'psk',4,'nondiff')).^2;
semilogy(simParams.EbNoVec,qpsk22BLERTh,':*')
hold on
semilogy(simParams.EbNoVec,qpsk88BLER,'--*')
qpsk88BLERTh = 1-(1-berawgn(simParams.EbNoVec,'psk',4,'nondiff')).^8;
semilogy(simParams.EbNoVec,qpsk88BLERTh,':o')
semilogy(simParams.EbNoVec,ae22eBLER,'-o')
semilogy(simParams.EbNoVec,ae44eBLER,'-d')
semilogy(simParams.EbNoVec,ae88eBLER,'-s')
hold off
ylim([1e-5 1])
grid on
xlabel('E_b/N_o (dB)')
ylabel('BLER')
legend('Uncoded QPSK (2,2)','Uncoded QPSK (8,8)','Theoretical Uncoded QPSK (8,8)',...
  'Autoencoder (2,2)','Autoencoder (4,4)','Autoencoder (8,8)','Location','southwest')
title('BLER performance of R=1 Autoencoders')
% Bit error rate of QPSK is the same for both (8,8) and (2,2) cases. However, the BLER depends on the block length, , and gets worse as  increases as given by . As expected, BLER performance of (8,8) QPSK is worse than the (2,2) QPSK system. The BLER performance of (2,2) autoencoder matches the BLER performance of (2,2) QPSK. On the other hand, (4,4) and (8,8) autoencoders optimize the channel coder and the constellation jointly to obtain a coding gain with respect to the corresponding uncoded QPSK systems. 

% Effect of Training Eb/No on BLER Performance
% Train the (7,4) autoencoder with energy normalization under different  values and compare the BLER performance.
n = 7;
k = 4;
normalization = 'Energy';

EbNoVec = 1:3:10;
if trainNow
  for EbNoIdx = 1:length(EbNoVec)
    SNR_dB = EbNoVec(EbNoIdx);
    [txNetVec{EbNoIdx},rxNetVec{EbNoIdx},infoVec{EbNoIdx},trainedNetVec{EbNoIdx}] = ...
      helperAEWTrainWirelessAutoencoder(n,k,normalization,SNR_dB);
    BLERVec{EbNoIdx} = helperAEWAutoencoderBLER(txNetVec{EbNoIdx},rxNetVec{EbNoIdx},simParams);
  end
else
  load ae74TrainedEbNo1to10 BLERVec trainParams simParams txNetVec rxNetVec infoVec trainedNetVec EbNoVec
end
% Plot the BLER performance together with theoretical upper bound for hard decision decoded Hamming (7,4) code and simulated BLER of maximum likelihood decoded (MLD) Hamming (7,4) code. The BLER performance of the (7,4) autoencoder gets closer to the Hamming (7,4) code with MLD as the training  decreases from 10 dB to 1 dB, at which point it almost matches the MLD Hamming (7,4) code. 
berHamming = bercoding(simParams.EbNoVec,'hamming','hard',7);
blerHamming = 1-(1-berHamming).^7;
load codedBLERResults hammingML74BLER
figure
semilogy(simParams.EbNoVec,blerHamming,':k')
hold on
linespec = {'-*','-d','-o','-s',};
for EbNoIdx=length(EbNoVec):-1:1
  semilogy(simParams.EbNoVec,BLERVec{EbNoIdx},linespec{EbNoIdx})
end
semilogy(simParams.EbNoVec,hammingML74BLER,'--vk')
hold off
ylim([1e-5 1])
grid on
xlabel('E_b/N_o (dB)')
ylabel('BLER')
legend('(7,4) Hamming HDD Upper','(7,4) AE - Eb/No=10','(7,4) AE - Eb/No=7',...
  '(7,4) AE - Eb/No=4','(7,4) AE - Eb/No=1','Hamming (7,4) MLD','location','southwest')

% Conclusions and Further Exploration
% The BLER results show that it is possible for autoencoders to learn joint coding and modulation schemes in an unsupervised way. It is even possible to train an autoencoder with R=1 to obtain a coding gain as compared to traditional methods. The example also shows the effect of hyperparameters such as  on the BLER performance. 
% The results are obtained using the following default settings for training and BLER simulations:
trainParams.Plots = 'none';
trainParams.Verbose = false;
trainParams.MaxEpochs = 15;
trainParams.InitialLearnRate = 0.01;
trainParams.LearnRateSchedule = 'piecewise';
trainParams.LearnRateDropPeriod = 10;
trainParams.LearnRateDropFactor = 0.1;
trainParams.MiniBatchSize = 20*2^k;

simParams.EbNoVec = -2:0.5:8;
simParams.MinNumErrors = 100;
simParams.MaxNumFrames = 300;
simParams.NumSymbolsPerFrame = 10000;
simParams.SignalPower = 1;


% References
% [1] T. O’Shea and J. Hoydis, "An Introduction to Deep Learning for the Physical Layer," in IEEE Transactions on Cognitive Communications and Networking, vol. 3, no. 4, pp. 563-575, Dec. 2017, doi: 10.1109/TCCN.2017.2758370.
% [2] S. Dörner, S. Cammerer, J. Hoydis and S. t. Brink, "Deep Learning Based Communication Over the Air," in IEEE Journal of Selected Topics in Signal Processing, vol. 12, no. 1, pp. 132-143, Feb. 2018, doi: 10.1109/JSTSP.2017.2784180.
% Copyright 2020 The MathWorks, Inc.
