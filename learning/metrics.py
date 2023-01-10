from sklearn import metrics as met
import torch
import numpy as np

def accuracy(y_hat, y):
    assert y_hat.shape == y.shape, "y_hat and y are not the same! {} {}".format(y_hat.shape, y.shape)
    y_hat = y_hat.cpu().detach()
    y = y.cpu()
    mask = (y_hat > 0.5).float()
    correct = (mask == y).float()
    n_total = y.shape[0]
    n_correct = torch.sum(correct)
    acc = n_correct / n_total
    assert acc <= 1.0, "Accuracy is greater 1! n_correct {}, total {}".format(n_correct, n_total)
    return acc


class BinaryClassMetrics:
    def __init__(self):
        """Constructor.
        """
        pass


    def set_data(self, pProbabilities, pLabels):
        """Method to set the data.
​
        Args:
            pPredictions (numpy array nxc): Vector with the predictions.
            pLabels (numpy array nxc): Vector with the ground truth.
        """

        # Store the data.
        if isinstance(pProbabilities, torch.Tensor):
            pProbabilities = pProbabilities.detach().cpu().numpy()
        if isinstance(pLabels, torch.Tensor):
            pLabels = pLabels.detach().cpu().numpy()
        self.probabilities_ = pProbabilities
        self.labels_ = pLabels.astype(np.float64)
        self.invLabels_ = 1.0 - self.labels_


    def compute_predictions(self, pThreshold):
        """Method to compute the predictions.
​
        Args:
            pThreshold (float): Threshold used to divide the two classes.
        """
        self.predictions_ = np.zeros_like(self.probabilities_, dtype=np.float64)
        self.predictions_[self.probabilities_ >= pThreshold] = 1.0
        self.invPredictions_ = 1.0 - self.predictions_


    def compute_accuracy(self, pThreshold=0.5):
        """Method to compute the accuracy. Total correct predictions.
​        Args:
            pThreshold (float): Threshold used to divide the two classes.
        Returns:
            (float): Accuracy.
        """
        self.compute_predictions(pThreshold)
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)


    @property
    def TP(self):
        return np.sum(self.labels_ * self.predictions_)

    @property
    def TN(self):
        return np.sum(self.invLabels_ * self.invPredictions_)

    @property
    def FP(self):
        return np.sum(self.invLabels_ * self.predictions_)

    @property
    def FN(self):
        return np.sum(self.labels_ * self.invPredictions_)


    def compute_precision(self):
        """Method to compute the precision of the predictions.
​
        Returns:
            (float): Precision.
        """
        if self.TP <= 0.0: return 0.0
        return self.TP / (self.TP + self.FP)


    def compute_recall(self):
        """Method to compute the recall of the predictions.
​
        Returns:
            (float): Recall.
        """
        if self.TP <= 0.0: return 0.0
        return self.TP / (self.TP + self.FN)


    def compute_specificity(self):
        """Method to compute the specificity of the predictions.
​
        Return:
            (float): Specificity.
            (numpy array c): Specificity per class.
        """
    
        return self.TN / (self.TN + self.FP)

    def compute_sensitivity(self):
        """Method to compute the specificity of the predictions.
​
        Return:
            (float): Specificity.
            (numpy array c): Specificity per class.
        """
    
        return self.compute_recall()

    def compute_TPR(self):
        return self.compute_sensitivity()

    def compute_TNR(self):
        return self.compute_specificity()

    def compute_FPR(self):
        return 1.0 - self.compute_specificity()

    def compute_FNR(self):
        return self.FN / (self.TP + self.FN) 

    def compute_fscore(self):
        """Method compute the fscore.
​
        Returns:
            (float): FScore.
        """

        precision = self.compute_precision()
        recall = self.compute_recall()
        sumDiv = precision+recall
        if sumDiv > 0.0:
            fscore = 2.0*((precision*recall)/sumDiv)
        else:
            fscore = 0.0
        return fscore


    def compute_IoU(self):
        """Method to compute the intersection over union.
​
        Returns:
            (float): Intersection over union.
        """
        
        intersection = self.labels_*self.predictions_
        union = np.sum(np.clip(self.labels_+self.predictions_, 0.0, 1.0))
        if union > 0.0:
            return np.sum(intersection)/union
        else:
            return 1.0

    def compute_ROC(self, pNumSteps):
        roc = np.array([])
        for thIter in range(pNumSteps + 1):
            curThreshold = float(thIter)/float(pNumSteps)
            self.compute_predictions(curThreshold)
            roc = np.append(roc, [self.compute_FPR(), self.compute_TPR()])
        return roc.reshape(-1, 2)

    

    def compute_PR(self, pNumSteps):
        pr = np.array([])
        for thIter in range(pNumSteps + 1):
            curThreshold = float(thIter)/float(pNumSteps)
            self.compute_predictions(curThreshold)
            pr = np.append(pr, [self.compute_recall(), self.compute_precision()])
        pr = pr.reshape(-1, 2)
        return pr.reshape(-1, 2)


    def compute_area_under_curve(self, pNumSteps, curve):
        assert curve in ["ROC", "PR"], "Please provide a curve from: Receiver Characteristic Operator (ROC) or Precision Recall (PR)"        
        if curve == "PR":
            prec, rec, _ = met.precision_recall_curve(self.labels_, self.probabilities_)
            accumulate = met.auc(rec, prec)
        elif curve == "ROC":
            accumulate = met.roc_auc_score(self.labels_, self.probabilities_)
        return accumulate

    def compute_AUPR(self, pNumSteps=100):
        return self.compute_area_under_curve(pNumSteps, curve="PR")

    def compute_AUC(self, pNumSteps=100):
        return self.compute_area_under_curve(pNumSteps, curve="ROC")

if __name__ == '__main__':
    import torch
    
    y = np.array([1,0,0,1,1,1,0,1,1,1])
    y_hat = np.array([0.98, 0.67, 0.58, 0.78, 0.85, 0.86, 0.79, 0.39, 0.82, 0.86])
    
    
    y     = np.random.randint(0,2,100) 
    y_hat = np.random.uniform(0,1,100) 


    metric = BinaryClassMetrics()
    metric.set_data(y_hat, y)
    
    print("Acc = {}".format(metric.compute_accuracy()))
    print("AUC = {}".format(metric.compute_AUC()))
    print("AUPR = {}".format(metric.compute_AUPR()))
    