cd ~/Desktop/initiationIA_tp4
source venv/bin/activate


pour activer l'environnement


full aléatoire
python play.py -r frozen_lake8
python play.py -r taxi
python play.py -r cart_pole


entrainement 
python trainAI.py taxi -n 60000
etc

résultat
python play.py -i cart_pole


============================================
RESULTATS


1. Frozen Lake 4x4

Nombre d’itérations : 20 000

Résultat :

L’agent gagne presque toujours. Il apprend rapidement à éviter les trous et à atteindre la case d’arrivée.

Test avec play.py -i frozen_lake

Courbe d’apprentissage : agent-frozen_lake.png

2. Frozen Lake 8x8

Nombre d’itérations : 20 000

Résultat :

Malgré la taille plus grande du plateau, l’agent gagne presque toujours après l’apprentissage.

Test avec play.py -i frozen_lake8

Courbe d’apprentissage : agent_frozen_lake.png

3. Taxi

Nombre d’itérations : 80 000

Résultat :

L’agent réussit la mission presque toujours : il va chercher le passager, le transporte à destination et le dépose correctement.

Test avec play.py -i taxi

Courbe d’apprentissage : agent-taxi.png

4. CartPole

Nombre d’itérations : 300 000

Résultat :

Le poteau tient moyennement longtemps. L’agent parvient à stabiliser le système pendant un certain temps.
Test avec play.py -i cart_pole

Courbe d’apprentissage : agent-cart_pole.png

