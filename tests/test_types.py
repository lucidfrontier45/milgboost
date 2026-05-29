import numpy as np

from milgboost.types import (
    Bag,
    LabeledBag,
    arrays_to_bags,
    arrays_to_labeled_bags,
    bags_to_arrays,
    labeled_bags_to_arrays,
)


class TestBag:
    def test_bag_creation(self) -> None:
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        bag = Bag(features=features)
        assert np.array_equal(bag.features, features)

    def test_bag_is_dataclass(self) -> None:
        bag = Bag(features=np.array([1.0, 2.0]))
        assert Bag.__dataclass_fields__ is not None


class TestLabeledBag:
    def test_labeled_bag_creation(self) -> None:
        features = np.array([[1.0, 2.0]])
        bag = LabeledBag(features=features, label=1)
        assert np.array_equal(bag.features, features)
        assert bag.label == 1

    def test_labeled_bag_inherits_bag(self) -> None:
        assert issubclass(LabeledBag, Bag)


class TestArraysToBags:
    def test_two_bags(self) -> None:
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        z = np.array([0, 0, 1, 1])
        bags = arrays_to_bags(x, z)
        assert len(bags) == 2
        assert np.array_equal(bags[0].features, np.array([[1.0], [2.0]]))
        assert np.array_equal(bags[1].features, np.array([[3.0], [4.0]]))

    def test_unsorted_z(self) -> None:
        x = np.array([[1.0], [2.0], [3.0]])
        z = np.array([1, 0, 1])
        bags = arrays_to_bags(x, z)
        assert len(bags) == 2
        assert np.array_equal(bags[0].features, np.array([[2.0]]))
        assert np.array_equal(bags[1].features, np.array([[1.0], [3.0]]))

    def test_single_instance_per_bag(self) -> None:
        x = np.array([[1.0], [2.0], [3.0]])
        z = np.array([0, 1, 2])
        bags = arrays_to_bags(x, z)
        assert len(bags) == 3
        for i, bag in enumerate(bags):
            assert bag.features.shape == (1, 1)
            assert bag.features[0, 0] == float(i + 1)

    def test_roundtrip(self) -> None:
        x_orig = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        z_orig = np.array([0, 0, 1])
        bags = arrays_to_bags(x_orig, z_orig)
        x_back, z_back = bags_to_arrays(bags)
        assert np.array_equal(x_orig, x_back)
        assert np.array_equal(z_orig, z_back)


class TestBagsToArrays:
    def test_empty_list(self) -> None:
        x, z = bags_to_arrays([])
        assert x.shape == (0, 0)
        assert z.shape == (0,)

    def test_single_bag(self) -> None:
        bags = [Bag(features=np.array([[1.0], [2.0]]))]
        x, z = bags_to_arrays(bags)
        assert np.array_equal(x, np.array([[1.0], [2.0]]))
        assert np.array_equal(z, np.array([0, 0]))

    def test_multiple_bags_varying_sizes(self) -> None:
        bags = [
            Bag(features=np.array([[1.0, 2.0]])),
            Bag(features=np.array([[3.0, 4.0], [5.0, 6.0]])),
        ]
        x, z = bags_to_arrays(bags)
        expected_x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected_z = np.array([0, 1, 1])
        assert np.array_equal(x, expected_x)
        assert np.array_equal(z, expected_z)


class TestArraysToLabeledBags:
    def test_two_bags(self) -> None:
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 1])
        z = np.array([0, 0, 1, 1])
        bags = arrays_to_labeled_bags(x, y, z)
        assert len(bags) == 2
        assert bags[0].label == 0
        assert bags[1].label == 1
        assert np.array_equal(bags[0].features, np.array([[1.0], [2.0]]))
        assert np.array_equal(bags[1].features, np.array([[3.0], [4.0]]))

    def test_unsorted_z(self) -> None:
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1, 0])
        z = np.array([1, 0, 1])
        bags = arrays_to_labeled_bags(x, y, z)
        assert len(bags) == 2
        assert bags[0].label == 1
        assert bags[1].label == 0
        assert np.array_equal(bags[0].features, np.array([[2.0]]))
        assert np.array_equal(bags[1].features, np.array([[1.0], [3.0]]))

    def test_roundtrip(self) -> None:
        x_orig = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_orig = np.array([0, 1])
        z_orig = np.array([0, 0, 1, 1])
        bags = arrays_to_labeled_bags(x_orig, y_orig, z_orig)
        x_back, y_back, z_back = labeled_bags_to_arrays(bags)
        assert np.array_equal(x_orig, x_back)
        assert np.array_equal(y_orig, y_back)
        assert np.array_equal(z_orig, z_back)


class TestLabeledBagsToArrays:
    def test_empty_list(self) -> None:
        x, y, z = labeled_bags_to_arrays([])
        assert x.shape == (0, 0)
        assert y.shape == (0,)
        assert z.shape == (0,)

    def test_single_bag(self) -> None:
        bags = [LabeledBag(features=np.array([[1.0], [2.0]]), label=1)]
        x, y, z = labeled_bags_to_arrays(bags)
        assert np.array_equal(x, np.array([[1.0], [2.0]]))
        assert np.array_equal(y, np.array([1]))
        assert np.array_equal(z, np.array([0, 0]))

    def test_multiple_bags_varying_sizes(self) -> None:
        bags = [
            LabeledBag(features=np.array([[1.0]]), label=0),
            LabeledBag(features=np.array([[2.0], [3.0]]), label=1),
        ]
        x, y, z = labeled_bags_to_arrays(bags)
        expected_x = np.array([[1.0], [2.0], [3.0]])
        expected_y = np.array([0, 1])
        expected_z = np.array([0, 1, 1])
        assert np.array_equal(x, expected_x)
        assert np.array_equal(y, expected_y)
        assert np.array_equal(z, expected_z)
