#!/usr/bin/env bash
MASTER=192.168.31.160
SLAVES=(192.168.31.227 192.168.31.236 192.168.0.59)

echo "[MASTER] Копируем install_slave.sh на все узлы"
for host in "${SLAVES[@]}"; do
  scp install_slave.sh "$host:/tmp/"
done

echo "[MASTER] Запускаем установку параллельно"
for host in "${SLAVES[@]}"; do
  ssh -o "StrictHostKeyChecking=no" "$host" "bash /tmp/install_slave.sh" &
done
wait
echo "[MASTER] Все slave-узлы инициализированы. Дайте им перезагрузиться."
