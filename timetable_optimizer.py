import json
from ortools.sat.python import cp_model


def solve_timetable():
    # 1. Chargement des données
    try:
        with open("subjects.json", "r", encoding="utf-8") as f:
            subjects_data = json.load(f)
        with open("rooms.json", "r", encoding="utf-8") as f:
            rooms_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Erreur : Le fichier {e.filename} est introuvable.")
        return

    model = cp_model.CpModel()

    # Paramètres temporels
    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
    periods = [1, 2, 3, 4, 5]
    # Pondération stricte de l'énoncé : w5 > w4 > w3 > w2 > w1
    weights = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160}

    rooms = rooms_data.get("Informatique", [])

    all_courses = []
    teachers = set()
    levels = set()

    # Extraction de TOUTES les données (S1 et S2 pour tous les niveaux)
    for level_key, semesters in subjects_data["niveau"].items():
        levels.add(level_key)
        for sem_key in ["s1", "s2"]:
            if sem_key in semesters:
                for sub in semesters[sem_key]["subjects"]:
                    if not sub.get("code"):
                        continue

                    lecturer_list = sub.get("Course Lecturer", ["Inconnu"])
                    teacher_name = " ".join([n for n in lecturer_list if n]).strip()

                    # On crée un objet cours unique (car un code peut apparaître 2 fois)
                    all_courses.append(
                        {
                            "unique_id": f"{level_key}_{sem_key}_{sub['code']}",
                            "code": sub["code"],
                            "name": sub["name"],
                            "teacher": teacher_name,
                            "level": level_key,
                            "semester": sem_key,
                        }
                    )
                    teachers.add(teacher_name)

    # 2. Variables de décision
    # x[(index_cours, index_salle, jour, periode)]
    x = {}
    for c_idx in range(len(all_courses)):
        for r_idx in range(len(rooms)):
            for d in range(len(days)):
                for p in periods:
                    x[(c_idx, r_idx, d, p)] = model.NewBoolVar(
                        f"x_c{c_idx}_r{r_idx}_d{d}_p{p}"
                    )

    # 3. Contraintes

    # C1 : Chaque cours (S1 et S2) doit être planifié exactement une fois par semaine
    for c_idx in range(len(all_courses)):
        model.Add(
            sum(
                x[(c_idx, r_idx, d, p)]
                for r_idx in range(len(rooms))
                for d in range(len(days))
                for p in periods
            )
            == 1
        )

    # C2 : Une salle = Un seul cours à la fois
    for r_idx in range(len(rooms)):
        for d in range(len(days)):
            for p in periods:
                model.AddAtMostOne(
                    x[(c_idx, r_idx, d, p)] for c_idx in range(len(all_courses))
                )

    # C3 : Un enseignant = Un seul cours à la fois (tous niveaux confondus)
    for t in teachers:
        for d in range(len(days)):
            for p in periods:
                model.AddAtMostOne(
                    x[(c_idx, r_idx, d, p)]
                    for c_idx, c in enumerate(all_courses)
                    if c["teacher"] == t
                    for r_idx in range(len(rooms))
                )

    # C4 : Une classe (Niveau) = Un seul cours à la fois (S1 et S2 cumulés)
    for l in levels:
        for d in range(len(days)):
            for p in periods:
                model.AddAtMostOne(
                    x[(c_idx, r_idx, d, p)]
                    for c_idx, c in enumerate(all_courses)
                    if c["level"] == l
                    for r_idx in range(len(rooms))
                )

    # 4. Fonction Objectif : Respecter w5 > w4 > ... > w1
    objective_terms = []
    for (c_idx, r_idx, d, p), var in x.items():
        objective_terms.append(var * weights[p])
    model.Maximize(sum(objective_terms))

    # 5. Résolution
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Emploi du temps Global généré avec succès.\n")

        # Liste des semestres pour l'affichage ordonné
        for target_sem in ["s1", "s2"]:
            header = (
                "PREMIER SEMESTRE (S1)"
                if target_sem == "s1"
                else "SECOND SEMESTRE (S2)"
            )
            print("=" * 50)
            print(f"      {header}")
            print("=" * 50 + "\n")

            for d_idx, day_name in enumerate(days):
                # On vérifie s'il y a des cours pour ce semestre ce jour-là
                has_courses_today = any(
                    solver.Value(x[(c_idx, r_idx, d_idx, p)]) == 1
                    for c_idx, c in enumerate(all_courses)
                    if c["semester"] == target_sem
                    for r_idx in range(len(rooms))
                    for p in periods
                )

                if has_courses_today:
                    print(f"--- {day_name.upper()} ---")
                    for p in periods:
                        for c_idx, c in enumerate(all_courses):
                            if c["semester"] == target_sem:
                                for r_idx, r in enumerate(rooms):
                                    if solver.Value(x[(c_idx, r_idx, d_idx, p)]) == 1:
                                        print(
                                            f"P{p} | Niv {c['level']} | {c['code']} - {c['name']}"
                                        )
                                        print(
                                            f"     Salle: {r['num']} | Prof: {c['teacher']}"
                                        )
                                        print("-" * 30)
                    print()
            print("\n")
    else:
        print(
            "Erreur : Aucune solution n'a pu être trouvée avec les contraintes actuelles."
        )


if __name__ == "__main__":
    solve_timetable()
