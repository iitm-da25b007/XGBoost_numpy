import numpy as np

# Default parameters of model
lr = 0.1
gamma = 0.0
Lambda = 1.0
max_depth = 3
min_child_weight = 1.0
n_estimators = 100

class Node:
    def __init__(
        self,
        value=None,
        feature=None,
        threshold=None,
        left=None,
        right=None,
    ):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.feature is None


class XGBoostScratch:
    def __init__(self, n_estimators=100, lr=0.1, gamma=0.0, lambda_param=1.0, max_depth=3, min_child_weight=1.0):
        self.trees = []
        self.n_estimators = n_estimators
        self.lr = lr
        self.gamma = gamma
        self.lambda_param = lambda_param
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.base_score = 0.0

    def _build_tree(self, X, grad, hess, depth):
        G = np.sum(grad)
        H = np.sum(hess)

        # Optimal prediction for this node if it becomes a leaf
        leaf_value = -G / (H + self.lambda_param)

        # Stop condition for growth
        if depth >= self.max_depth or len(grad) < 2:
            return Node(value=leaf_value)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):
            order = np.argsort(X[:, feature])
            sorted_values = X[order, feature]
            sorted_grad = grad[order]
            sorted_hess = hess[order]

            grad_sum = np.cumsum(sorted_grad)
            hess_sum = np.cumsum(sorted_hess)

            # Candidate positions where adjacent feature values differ
            positions = np.flatnonzero(sorted_values[:-1] != sorted_values[1:])

            if positions.size == 0:
                continue

            left_grad = grad_sum[positions]
            left_hess = hess_sum[positions]
            right_grad = G - left_grad
            right_hess = H - left_hess

            # Enforce minimum child weight
            valid = ((left_hess >= self.min_child_weight)
                    & (right_hess >= self.min_child_weight))

            if not np.any(valid):
                continue

            # Calculate split gains for all valid thresholds at once
            gains = 0.5 * (
                left_grad**2 / (left_hess + self.lambda_param)
                + right_grad**2 / (right_hess + self.lambda_param)
                - G**2 / (H + self.lambda_param)
            ) - self.gamma

            gains[~valid] = -np.inf

            best_index = np.argmax(gains)
            gain = gains[best_index]

            if gain > best_gain:
                position = positions[best_index]

                best_gain = gain
                best_feature = feature
                best_threshold = (sorted_values[position] + sorted_values[position + 1]) / 2.0

        # No useful split exists
        if best_feature is None:
            return Node(value=leaf_value)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_child = self._build_tree(
            X[left_mask],
            grad[left_mask],
            hess[left_mask],
            depth + 1,
        )
        right_child = self._build_tree(
            X[right_mask],
            grad[right_mask],
            hess[right_mask],
            depth + 1,
        )

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
        )

    def _predict_tree(self, tree, X):
        X = np.asarray(X, dtype=float)
        predictions = np.empty(X.shape[0], dtype=float)
        stack = [(tree, np.arange(X.shape[0]))]

        while stack:
            node, indices = stack.pop()
            if indices.size == 0:
                continue
            if node.is_leaf():
                predictions[indices] = node.value
                continue

            go_left = X[indices, node.feature] <= node.threshold
            stack.append((node.right, indices[~go_left]))
            stack.append((node.left, indices[go_left]))

        return predictions

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(y) == 0:
            raise ValueError("Training data cannot be empty")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        # Baseline prediction(i.e. mean of y)
        self.base_score = np.mean(y)
        predictions = np.full(len(y), self.base_score)

        self.trees = []

        for _ in range(self.n_estimators):
            # First and second derivatives of  the obj (0.5 * (y - y_pred)^2)
            grad = predictions - y
            hess = np.ones_like(grad)

            tree = self._build_tree(X, grad, hess, depth=0)
            tree_update = self._predict_tree(tree, X)

            predictions += self.lr * tree_update
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = np.full(len(X), self.base_score)
        for tree in self.trees:
            predictions += self.lr * self._predict_tree(tree, X)

        return predictions