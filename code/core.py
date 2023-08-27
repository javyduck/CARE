import torch
from torch.nn.functional import softmax, sigmoid
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import norm, binom_test
from math import ceil
from statsmodels.stats.proportion import proportion_confint

class Smooth(object):

    ABSTAIN = -1

    def __init__(self, gcn_model: torch.nn.Module, num_classes: int, dataset: str, sigma: float):
        self.gcn_model = gcn_model
        self.num_classes = num_classes
        self.dataset = dataset
        self.sigma = sigma

    def base_classifier(self, batch: torch.tensor):
        if self.dataset == 'AWA':
            return self._base_classifier_AWA(batch)
        elif self.dataset == 'word50_word':
            return self._base_classifier_word50(batch)
        elif self.dataset == 'stop_sign':
            return self._base_classifier_stop_sign(batch)
        else:
            ### need to fix the edge_index here.
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    
    def _base_classifier_AWA(self, batch: torch.tensor) -> torch.tensor:
        noise_sd = self.sigma  
        main_path = [f'saved_models/AwA/noise_sd_{self.noise_sd:.2f}/main.pth.tar']
        all_path=[f'saved_models/AwA/noise_sd_{self.noise_sd:.2f}/main.pth.tar']
        all_path.extend([f'saved_models/AwA/noise_sd_{self.noise_sd:.2f}/attribute_{i}.pth.tar' for i in range(85)])
        all_path.extend([f'saved_models/AwA/noise_sd_{self.noise_sd:.2f}/hierarchy_{i}.pth.tar' for i in range(28)])
        confidence = torch.FloatTensor()

        for model_id, path in enumerate(all_path):
            checkpoint = torch.load(path)
            if model_id == 0:
                model = get_architecture('resnet50', 'AWA', classes=50)
                m = Softmax(dim=1)
            else:
                model = get_architecture('resnet50', 'AWA', classes=1)
                m = Sigmoid(dim=1)

            model.load_state_dict(checkpoint['state_dict'])
            model.eval()

            with torch.no_grad():
                logits = model(batch.cuda())
                new_confidence = m(logits).detach().cpu()
                confidence = torch.cat((confidence, new_confidence), dim=1)

        return self.gcn_model(confidence.unsqueeze(-1))[:,:self.num_classes]

    def _base_classifier_stop_sign(self, batch: torch.tensor) -> torch.tensor:
        noise_sd = self.sigma  
        all_path = [f'saved_models/stop_sign/noise_sd_{self.noise_sd:.2f}/{self.root}.pth.tar']
        all_path.extend([f'saved_models/stop_sign/noise_sd_{self.noise_sd:.2f}/attribute_{i}.pth.tar' for i in range(20)])
        
        # Initialize an empty tensor to store confidence values
        confidence = torch.FloatTensor()

        for model_id, path in enumerate(all_path):
            checkpoint = torch.load(path)
            if model_id == 0:
                model = get_architecture('neural', 'stop_sign')
            else:
                model = get_architecture('neural_attribute', 'stop_sign')
            m = Softmax(dim=1)

            model.load_state_dict(checkpoint['state_dict'])
            model.eval()

            with torch.no_grad():
                batch = batch.cuda()
                logits = model(batch)
                new_confidence = m(logits).detach().cpu()[:, [1 if model_id != 0 else slice(None)]]
                confidence = torch.cat((confidence, new_confidence), dim=1)

        return self.gcn_model(confidence.unsqueeze(-1))[:,:self.num_classes]
    
    def _base_classifier_word50(self, batch: torch.tensor) -> torch.tensor:
        noise_sd = self.sigma  # use your noise_sd
        data = batch.clone().reshape(-1, 5 * 28 * 28)

        all_path = [f'saved_models/word50/main/main-1_noise_sd{noise_sd:.2f}.pth.tar']
        all_path.extend([f'saved_models/word50/extra/extra-{i}_noise_sd{noise_sd:.2f}.pth.tar' for i in range(1, 6)])

        # Initialize an empty tensor to store confidence values
        confidence = torch.FloatTensor()

        for model_id, path in enumerate(all_path):
            checkpoint = torch.load(path)
            if model_id == 0:
                x = data
                model = get_architecture('MLP', 'word50_word')
            else:
                x = data[:, 28 * 28 * (model_id - 1): 28 * 28 * model_id]
                model = get_architecture('MLP', 'word50_letter')

            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            m = Softmax(dim=1)

            with torch.no_grad():
                batch = x.cuda()
                logits = model(batch)
                new_confidence = m(logits).detach().cpu()
                confidence = torch.cat((confidence, new_confidence), dim=1)

        return self.gcn_model(confidence.unsqueeze(-1))[:,:self.num_classes]