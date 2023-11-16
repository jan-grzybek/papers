import torch
import torch.nn as nn
import random
from main import FamilyMember


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(24, 6, bias=False)
        self.fc1 = nn.Linear(12, 6, bias=False)
        self.fc2 = nn.Linear(12, 12, bias=False)
        self.fc3 = nn.Linear(12, 6, bias=False)
        self.fc4 = nn.Linear(6, 24, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, a, b):
        a = self.sig(self.fc0(a))
        b = self.sig(self.fc1(b))
        x = torch.cat((a, b), 0)
        x = self.sig(self.fc2(x))
        x = self.sig(self.fc3(x))
        x = self.sig(self.fc4(x))
        return x


def main():
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

    net = Net()
    criterion = nn.MSELoss()

    for s in range(1500):
        for language in ["en", "it"]:
            for i, t in enumerate(triples):
                if i in test_set[language]:
                    continue
                name_code, relationship = t[0], t[1]
                for name in all:
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

    output = net()
    # loss = criterion(output, target)
    print(net(torch.randn(24), torch.randn(12)))


if __name__ == "__main__":
    main()
