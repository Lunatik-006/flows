Federated Learning On Workstation Sharing

### Инструкция по запуску **F**ederated **L**earning **O**n **W**orkstation **S**haring

#1
Запускаем на **master**
(подготовка мастера)
```bash prepare_master.sh```
Затем ```sudo reboot```
#2
Запускаем на **master**
(Копируем и ставим всё на **slaves**)
```bash deploy_slaves.sh```
#3
Запускаем на **master**
(стартуем инференс)
```bash run_distributed.sh```

#4
