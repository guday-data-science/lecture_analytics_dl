# lecture_analytics_dl


```sh

CONTAINER ID        IMAGE                  COMMAND                  CREATED             STATUS              PORTS                                            NAMES
bef69c5d49e9        lecture_analytics_dl   "tini -g -- /bin/b..."   9 seconds ago       Up 8 seconds        0.0.0.0:6609->6006/tcp, 0.0.0.0:8809->8888/tcp   vigilant_leakey
d65f9883c682        lecture_analytics_dl   "tini -g -- /bin/b..."   10 seconds ago      Up 9 seconds        0.0.0.0:6608->6006/tcp, 0.0.0.0:8808->8888/tcp   elated_joliot
abb071ead75d        lecture_analytics_dl   "tini -g -- /bin/b..."   10 seconds ago      Up 9 seconds        0.0.0.0:6607->6006/tcp, 0.0.0.0:8807->8888/tcp   vibrant_leavitt
c5da3d20fb30        lecture_analytics_dl   "tini -g -- /bin/b..."   11 seconds ago      Up 10 seconds       0.0.0.0:6606->6006/tcp, 0.0.0.0:8806->8888/tcp   boring_bhabha
4e267c2819f3        lecture_analytics_dl   "tini -g -- /bin/b..."   12 seconds ago      Up 11 seconds       0.0.0.0:6605->6006/tcp, 0.0.0.0:8805->8888/tcp   laughing_tesla
06adf776eb51        lecture_analytics_dl   "tini -g -- /bin/b..."   12 seconds ago      Up 11 seconds       0.0.0.0:6604->6006/tcp, 0.0.0.0:8804->8888/tcp   heuristic_leakey
526ae89e7bc6        lecture_analytics_dl   "tini -g -- /bin/b..."   13 seconds ago      Up 12 seconds       0.0.0.0:6603->6006/tcp, 0.0.0.0:8803->8888/tcp   blissful_dubinsky
f1352960b69c        lecture_analytics_dl   "tini -g -- /bin/b..."   13 seconds ago      Up 12 seconds       0.0.0.0:6602->6006/tcp, 0.0.0.0:8802->8888/tcp   elegant_kilby
d73e60962b13        lecture_analytics_dl   "tini -g -- /bin/b..."   14 seconds ago      Up 13 seconds       0.0.0.0:6601->6006/tcp, 0.0.0.0:8801->8888/tcp   xenodochial_kowalevski
09848e1fddae        lecture_analytics_dl   "tini -g -- /bin/b..."   14 seconds ago      Up 13 seconds       0.0.0.0:6600->6006/tcp, 0.0.0.0:8800->8888/tcp   affectionate_payne

```


sample = pd.read_csv('1807_TOY_X.tar', nrows=5000)
sample = sample[sample['VEND_ID'] != 'VEND_ID']
sample.to_csv('sample.csv', index=False, header=True)
sample = pd.read_csv('sample.csv', dtype={i :'object' for i in range(5)})
