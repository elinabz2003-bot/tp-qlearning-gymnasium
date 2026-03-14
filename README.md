cd ~/Desktop/initiationIA_tp4
source venv/bin/activate

Fait par :  - Elina BAZZAZ ABKENAR
            - Clément BOUIX

pour activer l'environnement


full aléatoire
python3 play.py -r frozen_lake8
python3 play.py -r taxi
python3 play.py -r cart_pole


entrainement 
python3 trainAI.py frozen_lake -n 40000
python3 trainAI.py frozen_lake8 -n 100000
python3 trainAI.py taxi -n 60000
python3 trainAI.py cart_pole -n 20000

résultat de l'entrainement
python3 play.py frozen_lake --ai
python3 play.py frozen_lake8 --ai
python3 play.py taxi --ai
python3 play.py cart_pole --ai


============================================
RESULTATS


1. Frozen Lake 4x4

Nombre d’itérations : 40 000

Résultat :

L’agent gagne presque toujours. Il apprend rapidement à éviter les trous et à atteindre la case d’arrivée.

Test avec : python3 play.py frozen_lake --ai

Courbe d’apprentissage : agent-frozen_lake.png

2. Frozen Lake 8x8

Nombre d’itérations : 100 000

Résultat :

Malgré la taille plus grande du plateau, l’agent gagne presque toujours après l’apprentissage.

Test avec : python3 play.py frozen_lake8 --ai

Courbe d’apprentissage : agent_frozen_lake.png

3. Taxi

Nombre d’itérations : 60 000

Résultat :

L’agent réussit la mission presque toujours : il va chercher le passager, le transporte à destination et le dépose correctement.

Test avec : python3 play.py taxi --ai

Courbe d’apprentissage : agent-taxi.png

4. CartPole

Nombre d’itérations : 20 000

Résultat :

Le poteau tient moyennement longtemps. L’agent parvient à stabiliser le système pendant un certain temps.
Test avec : python3 play.py cart_pole --ai

Courbe d’apprentissage : agent-cart_pole.png

