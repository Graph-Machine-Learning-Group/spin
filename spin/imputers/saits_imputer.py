from tsl.imputers import Imputer


class SAITSImputer(Imputer):

    def shared_step(self, batch, mask):
        y = y_loss = batch.y
        y_hat = y_hat_loss = self.predict_batch(batch, preprocess=False,
                                                postprocess=not self.scale_target)

        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        y_hat_loss, y_loss, mask = self.trim_warm_up(y_hat_loss, y_loss, mask)

        if isinstance(y_hat_loss, (list, tuple)):
            imputation, predictions = y_hat_loss
            y_hat = y_hat[0]
        else:
            imputation, predictions = y_hat_loss, []

        # Imputation loss
        if self.training:
            injected_missing = batch.original_mask - batch.mask
            mask = batch.mask
            loss = self.loss_fn(imputation, y_loss, injected_missing)
        else:
            loss = 0

        # Reconstruction loss
        for pred in predictions:
            pred_loss = self.loss_fn(pred, y_loss, mask)
            loss += self.prediction_loss_weight * pred_loss / 3

        return y_hat.detach(), y, loss
