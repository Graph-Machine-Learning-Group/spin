from tsl.imputers import Imputer

from ..baselines import BRITS


class BRITSImputer(Imputer):

    def shared_step(self, batch, mask):
        y = y_loss = batch.y
        y_hat = y_hat_loss = self.predict_batch(batch, preprocess=False,
                                                postprocess=not self.scale_target)

        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        y_hat_loss, y_loss, mask = self.trim_warm_up(y_hat_loss, y_loss, mask)

        imputation, predictions = y_hat_loss
        imp_fwd, imp_bwd = predictions[:2]
        y_hat = y_hat[0]

        loss = sum([self.loss_fn(pred, y_loss, mask) for pred in predictions])
        loss += BRITS.consistency_loss(imp_fwd, imp_bwd)

        return y_hat.detach(), y, loss
