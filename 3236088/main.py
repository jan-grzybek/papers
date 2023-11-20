# http://www.cs.utoronto.ca/~hinton/absps/naturebp.pdf
# https://www.cs.toronto.edu/~hinton/absps/families.pdf
# Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors.
# Nature, 323(6088), 533–536. https://doi.org/10.1038/323533a0

import math
import time
import random
import argparse
from uuid import uuid4
from tqdm import tqdm

global_gradients = {}


class Op:
    def __init__(self):
        self._cache_uuid = None
        self._cache = None

    def _connect_upper(self, upper_unit):
        raise NotImplementedError

    def output(self, uuid=None):
        raise NotImplementedError

    def backprop(self, upper_unit, derivative=None, uuid=None):
        raise NotImplementedError

    def dump_gradients(self, uuid=None):
        raise NotImplementedError

    def update(self, eps, alpha, weight_decay=0., uuid=None):
        raise NotImplementedError


class Unit(Op):
    def __init__(self):
        super().__init__()
        self._connected_units = []
        self._accumulated_gradient_weights = []
        self._gradient_descents = []
        self._upper_units = []
        self._upper_derivatives = {}

    def _connect_upper(self, upper_unit):
        self._upper_units.append(upper_unit)

    def connect(self, unit, weight: float):
        self._connected_units.append([unit, weight])
        self._accumulated_gradient_weights.append(0.)
        self._gradient_descents.append(0.)
        unit._connect_upper(self)
        return self

    def output(self, uuid=None):
        if uuid is None:
            uuid = uuid4()
        elif uuid == self._cache_uuid and self._cache is not None:
            return self._cache
        total_input = 0.
        for unit, weight in self._connected_units:
            total_input += unit.output(uuid) * weight
        self._cache_uuid = uuid
        self._cache = 1 / (1 + math.e ** -total_input)
        return self._cache

    def backprop(self, upper_unit, derivative=1., uuid=None):
        if upper_unit is not None:
            assert issubclass(type(upper_unit), Op)
            assert upper_unit not in self._upper_derivatives.keys()
            self._upper_derivatives[upper_unit] = derivative
            if len(self._upper_derivatives) == len(self._upper_units):
                derivative = sum(self._upper_derivatives.values())
            else:
                return
        if uuid is None:
            uuid = uuid4()
        out = self.output(uuid)
        local_derivative = derivative * out * (1 - out)
        for i, x in enumerate(self._connected_units):
            input_unit, weight = x
            self._accumulated_gradient_weights[i] += local_derivative * input_unit.output(uuid)
            input_unit.backprop(self, local_derivative * weight, uuid)
        self._upper_derivatives = {}
        return out

    def dump_gradients(self, uuid=None):
        if uuid is None:
            uuid = uuid4()
        if uuid != self._cache_uuid:
            self._cache_uuid = uuid
            global global_gradients
            global_gradients[self] = {}
            for i, (input_unit, _) in enumerate(self._connected_units):
                global_gradients[self][i] = self._accumulated_gradient_weights[i]
                input_unit.dump_gradients(uuid)

    def update(self, eps, alpha, weight_decay=0., uuid=None):
        if uuid is None:
            uuid = uuid4()
        if uuid != self._cache_uuid:
            self._cache_uuid = uuid
            for i, x in enumerate(self._connected_units):
                input_unit, weight = x
                self._gradient_descents[i] = (eps * self._accumulated_gradient_weights[i]
                                              + alpha * self._gradient_descents[i])
                self._connected_units[i][1] -= self._gradient_descents[i]
                self._connected_units[i][1] *= (1. - weight_decay)
                self._accumulated_gradient_weights[i] = 0.
                input_unit.update(eps, alpha, weight_decay, uuid)


class InputUnit(Op):
    def __init__(self, value=None):
        super().__init__()
        self.value = value

    def _connect_upper(self, upper_unit):
        return

    def output(self, uuid=None) -> float:
        return self.value

    def backprop(self, upper_unit, derivative=1., uuid=None):
        return self.value

    def dump_gradients(self, uuid=None):
        return

    def update(self, eps, alpha, weight_decay=0., uuid=None):
        return


class Error(Op):
    def __init__(self, lower_threshold=None, upper_threshold=None):
        super().__init__()
        self._connected_units = []
        self._lower_threshold = lower_threshold
        self._upper_threshold = upper_threshold

    def connect(self, output_unit, reference_fn):
        self._connected_units.append((output_unit, reference_fn))

    def output(self, uuid=None):
        if uuid is None:
            uuid = uuid4()
        elif uuid == self._cache_uuid and self._cache is not None:
            return self._cache
        e = 0
        for output_unit, reference_fn in self._connected_units:
            out = output_unit.output(uuid)
            ref = reference_fn.output(uuid)
            if self._lower_threshold is not None:
                if out < self._lower_threshold and ref == 0:
                    e += 0
                    continue
            if self._upper_threshold is not None:
                if out > self._upper_threshold and ref == 1:
                    e += 0
                    continue
            e += (out - ref) ** 2
        self._cache_uuid = uuid
        self._cache = e / 2
        return self._cache

    def backprop(self, upper_unit=None, derivative=None, uuid=None):
        if uuid is None:
            uuid = uuid4()
        for output_unit, reference_fn in self._connected_units:
            out = output_unit.output(uuid)
            ref = reference_fn.output(uuid)
            if self._lower_threshold is not None:
                if out < self._lower_threshold and ref == 0:
                    output_unit.backprop(upper_unit, 0, uuid)
                    continue
            if self._upper_threshold is not None:
                if out > self._upper_threshold and ref == 1:
                    output_unit.backprop(upper_unit, 0, uuid)
                    continue
            output_unit.backprop(upper_unit, out - ref, uuid)
        return self.output(uuid)

    def dump_gradients(self, uuid=None):
        if uuid is None:
            uuid = uuid4()
        for output_unit, _ in self._connected_units:
            output_unit.dump_gradients(uuid)

    def update(self, eps, alpha, weight_decay=0., uuid=None):
        if uuid is None:
            uuid = uuid4()
        if uuid != self._cache_uuid:
            self._cache_uuid = uuid
            for output_unit, _ in self._connected_units:
                output_unit.update(eps, alpha, weight_decay, uuid)


def get_combinations(positions: int, values: list[int] = None, combinations: list[list[int]] = None):
    if positions == 0:
        return combinations
    if combinations is None:
        new_combinations = [[val] for val in values]
    else:
        new_combinations = []
        for combination in combinations:
            for val in values:
                new_combinations.append(combination + [val])
    return get_combinations(positions - 1, values, new_combinations)


class IsSymmetric(Op):
    def __init__(self):
        super().__init__()
        self._connected_units = []

    def connect(self, unit):
        self._connected_units.append(unit)
        return self

    def output(self, uuid=None) -> float:
        if uuid is None:
            uuid = uuid4()
        elif uuid == self._cache_uuid and self._cache is not None:
            return self._cache
        vector_size = len(self._connected_units)
        for i in range(vector_size // 2):
            if self._connected_units[i].output(uuid) != self._connected_units[vector_size - i - 1].output(uuid):
                self._cache = 0.
                return self._cache
        self._cache = 1.
        return self._cache


def mirror_symmetry(sweeps_count=1425, input_units_count=6):
    combinations = get_combinations(6, [0, 1])
    output = Unit().connect(InputUnit(1), random.uniform(-0.3, 0.3))  # bias
    hidden_units = [Unit().connect(InputUnit(1), random.uniform(-0.3, 0.3)) for _ in range(2)]
    input_units = [InputUnit() for _ in range(input_units_count)]
    for hidden_unit in hidden_units:
        output.connect(hidden_unit, random.uniform(-0.3, 0.3))
    for hidden_unit in hidden_units:
        for input_unit in input_units:
            hidden_unit.connect(input_unit, random.uniform(-0.3, 0.3))

    is_symmetric = IsSymmetric()
    for input_unit in input_units:
        is_symmetric.connect(input_unit)

    e = Error()
    e.connect(output, is_symmetric)
    for s in range(sweeps_count):
        total_error = 0.
        for c in combinations:
            for i, input_unit in enumerate(input_units):
                input_unit.value = c[i]
            total_error += e.backprop()
        print(f"Sweep {s + 1}/{sweeps_count}; total error: {total_error}")
        e.update(eps=0.1, alpha=0.9)

    print("\nTest:")
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for c in combinations:
        for i, input_unit in enumerate(input_units):
            input_unit.value = c[i]
        out = output.output()
        gt = is_symmetric.output()
        out_bool = out > 0.5
        gt_bool = gt == 1.0
        if out_bool and gt_bool:
            tp += 1
        elif out_bool and not gt_bool:
            fp += 1
        elif not out_bool and not gt_bool:
            tn += 1
        elif not out_bool and gt_bool:
            fn += 1
        else:
            raise AssertionError
        print(f"Combination: {c}; output: {out} [{out_bool}]; gt: {gt} [{gt_bool}]")
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print(f"\nPrecision: {100 * precision}%; recall: {100 * recall}%")
    except ZeroDivisionError:
        print("\nCan't calculate precision and recall. "
              "Probably unlucky weights initialization and stuck at poor local minima. Try again.")
        raise AssertionError


class IsRelative(Op):
    _name_codes = ['m00', 'f00', 'm01', 'f01', 'f10', 'm10', 'f11', 'm11', 'f12', 'm12', 'm20', 'f20']
    _possible_people_count = 2 * len(_name_codes)
    _relationships = ["father", "mother", "husband", "wife", "son", "daughter", "uncle", "aunt", "brother", "sister",
                      "nephew", "niece"]
    _possible_relationships_count = len(_relationships)

    def __init__(self, name_code, is_italian, family):
        super().__init__()
        assert name_code in self._name_codes
        self._name_code = name_code
        self._is_italian = is_italian
        self._family = family
        self._connected_units = []

    def connect(self, unit):
        self._connected_units.append(unit)
        return self

    def output(self, uuid=None) -> float:
        if uuid is None:
            uuid = uuid4()
        elif uuid == self._cache_uuid and self._cache is not None:
            return self._cache

        assert len(self._connected_units) == self._possible_relationships_count + self._possible_people_count

        person_0_code = None
        person_0_italian = False
        for i, unit in enumerate(self._connected_units[:self._possible_people_count]):
            if unit.value == 1:
                assert person_0_code is None
                if i >= len(self._name_codes):
                    person_0_italian = True
                person_0_code = self._name_codes[i % len(self._name_codes)]
        assert person_0_code is not None

        relationship = None
        for i, unit in enumerate(self._connected_units[self._possible_people_count:]):
            if unit.value == 1:
                assert relationship is None
                relationship = self._relationships[i]
        assert relationship is not None

        self._cache_uuid = uuid

        if person_0_italian and not self._is_italian:
            self._cache = 0.
        elif not person_0_italian and self._is_italian:
            self._cache = 0.
        else:
            self._cache = float(self._name_code in [
                m.name for m in self._family[person_0_code].match_relatives(relationship)])
        return self._cache


class FamilyMember:
    def __init__(self, name, is_male=True):
        self.name = name
        self.is_male = is_male
        self.spouse = None
        self.children = []
        self.parents = []
        self.siblings = []
        self.nephews = []
        self.uncles = []

    def match_relatives(self, relationship):
        relatives = []
        if relationship == "father":
            for parent in self.parents:
                if parent.is_male:
                    relatives.append(parent)
        elif relationship == "mother":
            for parent in self.parents:
                if not parent.is_male:
                    relatives.append(parent)
        elif relationship == "husband":
            if self.spouse is not None and self.spouse.is_male:
                relatives.append(self.spouse)
        elif relationship == "wife":
            if self.spouse is not None and not self.spouse.is_male:
                relatives.append(self.spouse)
        elif relationship == "son":
            for child in self.children:
                if child.is_male:
                    relatives.append(child)
        elif relationship == "daughter":
            for child in self.children:
                if not child.is_male:
                    relatives.append(child)
        elif relationship == "uncle":
            for uncle in self.uncles:
                if uncle.is_male:
                    relatives.append(uncle)
        elif relationship == "aunt":
            for uncle in self.uncles:
                if not uncle.is_male:
                    relatives.append(uncle)
        elif relationship == "brother":
            for sibling in self.siblings:
                if sibling.is_male:
                    relatives.append(sibling)
        elif relationship == "sister":
            for sibling in self.siblings:
                if not sibling.is_male:
                    relatives.append(sibling)
        elif relationship == "nephew":
            for nephew in self.nephews:
                if nephew.is_male:
                    relatives.append(nephew)
        elif relationship == "niece":
            for nephew in self.nephews:
                if not nephew.is_male:
                    relatives.append(nephew)
        else:
            raise AssertionError
        return relatives

    def is_married_to(self, member):
        if self.spouse is not None or member is self:
            return
        self.spouse = member
        for child in self.spouse.children:
            child.is_child_of(self)
        for nephew in self.spouse.nephews:
            nephew.is_nephew_to(self)
        member.is_married_to(self)

    def _is_uncle_of(self, member):
        if member in self.nephews or member is self:
            return
        self.nephews.append(member)
        for sibling in member.siblings:
            sibling._is_nephew_to(self)
        member._is_nephew_to(self)

    def _is_nephew_to(self, member):
        if member in self.uncles or member is self:
            return
        self.uncles.append(member)
        if member.spouse is not None:
            self._is_nephew_to(member.spouse)
        member._is_uncle_of(self)

    def _is_sibling_to(self, member):
        if member in self.siblings or member is self:
            return
        self.siblings.append(member)
        for parent in member.parents:
            parent._is_parent_of(self)
        for child in member.children:
            child._is_nephew_to(self)
        member._is_sibling_to(self)

    def _is_parent_of(self, member):
        if member in self.children or member is self:
            return
        self.children.append(member)
        for sibling in member.siblings:
            sibling.is_child_of(self)
        for child in self.children:
            child._is_sibling_to(member)
        if self.spouse is not None:
            self.spouse._is_parent_of(member)
        member.is_child_of(self)

    def is_child_of(self, member):
        if member in self.parents or member is self:
            return
        self.parents.append(member)
        for sibling in member.siblings:
            sibling._is_uncle_of(self)
        member._is_parent_of(self)
        if member.spouse is not None:
            member.spouse._is_parent_of(self)


def family_trees(sweeps_count=1500):
    # data
    relationships = ["father", "mother", "husband", "wife", "son", "daughter", "uncle", "aunt", "brother", "sister",
                     "nephew", "niece"]
    family = {
        "m00": FamilyMember("m00"),
        "f00": FamilyMember("f00", False),
        "m01": FamilyMember("m01"),
        "f01": FamilyMember("f01", False),
        "m10": FamilyMember("m10"),
        "f10": FamilyMember("f10", False),
        "m11": FamilyMember("m11"),
        "f11": FamilyMember("f11", False),
        "m12": FamilyMember("m12"),
        "f12": FamilyMember("f12", False),
        "m20": FamilyMember("m20"),
        "f20": FamilyMember("f20", False),
    }
    family["f00"].is_married_to(family["m00"])
    family["f01"].is_married_to(family["m01"])
    family["f10"].is_married_to(family["m10"])
    family["f11"].is_married_to(family["m11"])
    family["f12"].is_married_to(family["m12"])
    family["m10"].is_child_of(family["m00"])
    family["f11"].is_child_of(family["m00"])
    family["m11"].is_child_of(family["m01"])
    family["f12"].is_child_of(family["m01"])
    family["f20"].is_child_of(family["m11"])
    family["m20"].is_child_of(family["m11"])

    triples = []
    for relation in relationships:
        for person_0_name, m in family.items():
            for person_1 in m.match_relatives(relation):
                triples.append((person_0_name, relation, person_1.name))

    test_set = {"en": random.sample(range(0, len(triples)), 2),
                "it": random.sample(range(0, len(triples)), 2)}

    en_encodings_map = {"Christopher": "m00", "Penelope": "f00", "Andrew": "m01", "Christine": "f01", "Margaret": "f10",
                        "Arthur": "m10", "Victoria": "f11", "James": "m11", "Jennifer": "f12", "Charles": "m12",
                        "Colin": "m20", "Charlotte": "f20"}
    en_names_map = {val: key for key, val in en_encodings_map.items()}
    it_encodings_map = {"Roberto": "m00", "Maria": "f00", "Pierro": "m01", "Francesca": "f01", "Gina": "f10",
                        "Emilio": "m10", "Lucia": "f11", "Marco": "m11", "Angela": "f12", "Tomaso": "m12",
                        "Alfonso": "m20", "Sophia": "f20"}
    it_names_map = {val: key for key, val in it_encodings_map.items()}
    names_map = {"en": {"name2code": en_encodings_map, "code2name": en_names_map},
                 "it": {"name2code": it_encodings_map, "code2name": it_names_map}}
    all_names = list(en_encodings_map.keys()) + list(it_encodings_map.keys())

    # network
    person_units = {name: InputUnit() for name in all_names}
    relationship_units = {name: InputUnit() for name in relationships}
    person_hidden_units = [Unit() for _ in range(6)]
    for hidden_unit in person_hidden_units:
        for unit in person_units.values():
            hidden_unit.connect(unit, random.uniform(-0.3, 0.3))
    relationship_hidden_units = [Unit() for _ in range(6)]
    for hidden_unit in relationship_hidden_units:
        for unit in relationship_units.values():
            hidden_unit.connect(unit, random.uniform(-0.3, 0.3))
    central_layer = [Unit() for _ in range(12)]
    for hidden_unit_i in central_layer:
        for hidden_unit_j in person_hidden_units + relationship_hidden_units:
            hidden_unit_i.connect(hidden_unit_j, random.uniform(-0.3, 0.3))
    penultimate_layer = [Unit() for _ in range(6)]
    for hidden_unit_i in penultimate_layer:
        for hidden_unit_j in central_layer:
            hidden_unit_i.connect(hidden_unit_j, random.uniform(-0.3, 0.3))
    output_units = {name: Unit() for name in all_names}
    for unit in output_units.values():
        for hidden_unit in penultimate_layer:
            unit.connect(hidden_unit, random.uniform(-0.3, 0.3))

    references = []
    for language in ["en", "it"]:
        for name_code in names_map[language]["name2code"].values():
            reference = IsRelative(name_code, language == "it", family)
            for input_unit in person_units.values():
                reference.connect(input_unit)
            for input_unit in relationship_units.values():
                reference.connect(input_unit)
            references.append(reference)

    e = Error(0.2, 0.8)
    for unit, reference in zip(output_units.values(), references):
        e.connect(unit, reference)

    for s in range(sweeps_count):
        # train
        start = time.time()
        total_error = 0.
        for language in ["en", "it"]:
            for i, t in enumerate(triples):
                if i in test_set[language]:
                    continue
                name_code, relationship = t[0], t[1]
                for name, input_unit in person_units.items():
                    try:
                        input_unit.value = int(names_map[language]["name2code"][name] == name_code)
                    except KeyError:
                        input_unit.value = 0
                for name, input_unit in relationship_units.items():
                    input_unit.value = int(name == relationship)
                total_error += e.backprop()

        print(f"\nSweep {s + 1}/{sweeps_count}; total error: {total_error}; time: {round(time.time() - start, 2)} sec")
        if s < 20:
            e.update(eps=0.005, alpha=0.5, weight_decay=0.002)
        else:
            e.update(eps=0.01, alpha=0.9, weight_decay=0.002)

        # test
        if s % 10 == 0:
            indent = 2 * " "
            for cat in ["train", "test"]:
                for language in ["en", "it"]:
                    print(f"{indent}{cat} predictions [{language}]:")
                    if cat == "train":
                        idx = random.sample(range(0, len(triples)), 2)
                    else:
                        idx = test_set[language]
                    for i in idx:
                        t = triples[i]
                        name_code, relationship = t[0], t[1]
                        for name, input_unit in person_units.items():
                            try:
                                input_unit.value = int(names_map[language]["name2code"][name] == name_code)
                            except KeyError:
                                input_unit.value = 0
                        for name, input_unit in relationship_units.items():
                            input_unit.value = int(name == relationship)
                        x = {name: round(output.output(), 4) for name, output in output_units.items()}
                        sorted_outputs = sorted(x.items(), key=lambda item: item[1])
                        print(f"{2 * indent}{names_map[language]['code2name'][t[0]]}'s {t[1]} is: "
                              f"{[i for i in reversed(sorted_outputs[-3:])]}; least likely: {sorted_outputs[:3]}")


def check_gradients():
    class Dummy(Op):
        def __init__(self):
            super().__init__()
            self.value = random.choice([0, 1])

        def randomize(self):
            self.value = random.choice([0, 1])

        def output(self, uuid=None):
            return self.value

    # network
    input_units_0 = [InputUnit() for _ in range(24)]
    input_units_1 = [InputUnit() for _ in range(12)]
    hidden_units_0 = [Unit() for _ in range(6)]
    for hidden_unit in hidden_units_0:
        for unit in input_units_0:
            hidden_unit.connect(unit, random.uniform(-0.3, 0.3))
    hidden_units_1 = [Unit() for _ in range(6)]
    for hidden_unit in hidden_units_1:
        for unit in input_units_1:
            hidden_unit.connect(unit, random.uniform(-0.3, 0.3))
    central_layer = [Unit() for _ in range(12)]
    for hidden_unit_i in central_layer:
        for hidden_unit_j in hidden_units_0 + hidden_units_1:
            hidden_unit_i.connect(hidden_unit_j, random.uniform(-0.3, 0.3))
    penultimate_layer = [Unit() for _ in range(6)]
    for hidden_unit_i in penultimate_layer:
        for hidden_unit_j in central_layer:
            hidden_unit_i.connect(hidden_unit_j, random.uniform(-0.3, 0.3))
    output_units = [Unit() for _ in range(24)]
    for unit in output_units:
        for hidden_unit in penultimate_layer:
            unit.connect(hidden_unit, random.uniform(-0.3, 0.3))

    references = [Dummy() for _ in range(len(output_units))]

    e = Error(0.2, 0.8)
    for unit, reference in zip(output_units, references):
        e.connect(unit, reference)

    epsilon = 1e-4
    acc_est_grad = {}
    rounds = 100
    diffs = []
    for _ in tqdm(range(1, rounds+1)):
        for input_unit in input_units_0 + input_units_1:
            input_unit.value = random.choice([0, 1])
        for ref in references:
            ref.randomize()

        e.backprop()
        e.dump_gradients()

        diffs = []
        i = 0
        for unit_i in global_gradients:
            for unit_j in (input_units_0 + input_units_1 + hidden_units_0 + hidden_units_1 + central_layer +
                           penultimate_layer + output_units):
                if unit_i is unit_j:
                    if unit_i not in acc_est_grad.keys():
                        acc_est_grad[unit_i] = {}
                    for weight, grad in global_gradients[unit_i].items():
                        tmp = unit_i._connected_units[weight][1]
                        unit_i._connected_units[weight][1] += epsilon
                        x = e.output()
                        unit_i._connected_units[weight][1] -= 2 * epsilon
                        y = e.output()
                        unit_i._connected_units[weight][1] = tmp
                        est_grad = (x - y) / (2 * epsilon)
                        if weight not in acc_est_grad[unit_i].keys():
                            acc_est_grad[unit_i][weight] = 0.
                        acc_est_grad[unit_i][weight] += est_grad
                        diff = abs(grad - acc_est_grad[unit_i][weight])
                        diffs.append(diff)
                        assert diff < 1e-8
                    i += 1
        assert i == len(global_gradients)
    print(f"\nDelta on gradients accumulated over {rounds} backward passes:")
    print(f"  mean: {round(sum(diffs)/len(diffs), 12)}, min: {round(min(diffs), 12)}, max: {round(max(diffs), 12)}")
    print("\nGradient check successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', choices=["symmetry", "family", "check_grads"], required=True)
    args = parser.parse_args()
    if args.run == "symmetry":
        mirror_symmetry()
    elif args.run == "family":
        family_trees()
    elif args.run == "check_grads":
        check_gradients()
