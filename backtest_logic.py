import pandas as pd
import numpy as np
import requests
import pandas_ta as ta

def buscar_dados_api(ativo, timeframe, data_inicio, data_fim, api_key):
    """Busca dados históricos da API da Polygon.io."""
    print(f"Buscando dados para {ativo} de {data_inicio} a {data_fim}...")
    API_URL = f"https://api.polygon.io/v2/aggs/ticker/{ativo}/range/1/{timeframe}/{data_inicio}/{data_fim}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': api_key}
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if not data.get('results'):
            print("Nenhum dado retornado pela API.")
            return None
        df = pd.DataFrame(data['results'])
        df['datetime'] = pd.to_datetime(df['t'], unit='ms')
        df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
        df.set_index('datetime', inplace=True)
        print(f"Dados carregados com sucesso! Total de {len(df)} candles.")
        return df[['open', 'high', 'low', 'close', 'volume']]
    except requests.exceptions.RequestException as e:
        print(f"Erro ao chamar a API da Polygon: {e}")
        return None

def detectar_sinais(df, tipo_estrategia):
    """
    Detecta sinais de compra ou venda usando uma máquina de estados para a lógica refinada.
    """
    if df is None or len(df) < 200:
        print("Dados insuficientes para calcular todos os indicadores.")
        return pd.DataFrame()

    # --- 1. Cálculo de Indicadores ---
    df.ta.ema(length=9, append=True, col_names=('MME9',))
    df.ta.ema(length=20, append=True, col_names=('MME20',))
    df.ta.ema(length=50, append=True, col_names=('MME50',))
    df.ta.ema(length=200, append=True, col_names=('MME200',))
    df.ta.sma(length=200, append=True, col_names=('SMA200',))
    df.dropna(inplace=True) # Remove linhas sem dados de indicadores

    sinais = []
    estado = "PROCURANDO_TENDENCIA" # Estados possíveis: PROCURANDO_TENDENCIA, PROCURANDO_GATILHO

    for i in range(len(df)):
        candle_atual = df.iloc[i]
        
        # Define as condições de tendência para compra e venda
        tendencia_de_alta = candle_atual['MME9'] > candle_atual['MME20'] and candle_atual['MME20'] > candle_atual['MME50']
        tendencia_de_baixa = candle_atual['MME50'] > candle_atual['MME20'] and candle_atual['MME20'] > candle_atual['MME9']

        if estado == "PROCURANDO_TENDENCIA":
            if tipo_estrategia == 'compra' and tendencia_de_alta:
                # Se a mínima tocar ou passar abaixo da MME50, muda o estado
                if candle_atual['low'] <= candle_atual['MME50']:
                    estado = "PROCURANDO_GATILHO"
                    print(f"[{candle_atual.name.date()}] COMPRA: Tendência OK, toque na MME50. Procurando gatilho...")
            
            elif tipo_estrategia == 'venda' and tendencia_de_baixa:
                # Se a máxima tocar ou passar acima da MME50, muda o estado
                if candle_atual['high'] >= candle_atual['MME50']:
                    estado = "PROCURANDO_GATILHO"
                    print(f"[{candle_atual.name.date()}] VENDA: Tendência OK, toque na MME50. Procurando gatilho...")

        elif estado == "PROCURANDO_GATILHO":
            # Verifica se a tendência se mantém. Se não, volta a procurar.
            if (tipo_estrategia == 'compra' and not tendencia_de_alta) or \
               (tipo_estrategia == 'venda' and not tendencia_de_baixa):
                estado = "PROCURANDO_TENDENCIA"
                print(f"[{candle_atual.name.date()}] Tendência quebrou. Voltando a procurar.")
                continue

            # Procura pelo candle de força (gatilho)
            candle_de_forca = False
            if tipo_estrategia == 'compra':
                # Candle de força comprador que fecha acima da MME20
                if candle_atual['close'] > candle_atual['open'] and candle_atual['close'] > candle_atual['MME20']:
                    candle_de_forca = True
            else: # Venda
                # Candle de força vendedor que fecha abaixo da MME20
                if candle_atual['close'] < candle_atual['open'] and candle_atual['close'] < candle_atual['MME20']:
                    candle_de_forca = True

            if candle_de_forca:
                print(f"[{candle_atual.name.date()}] GATILHO ENCONTRADO!")
                entrada = candle_atual['close']
                
                if tipo_estrategia == 'compra':
                    stop = candle_atual['low']
                    risco = entrada - stop
                    # Alvo primário é a MME200, se não, 2x o risco
                    alvo = candle_atual['MME200'] if entrada < candle_atual['MME200'] else entrada + (2 * risco)
                else: # Venda
                    stop = candle_atual['high']
                    risco = stop - entrada
                    # Alvo primário é a MME200, se não, 2x o risco
                    alvo = candle_atual['MME200'] if entrada > candle_atual['MME200'] else entrada - (2 * risco)

                if risco > 0: # Evita trades com risco zero
                    sinais.append({
                        "data": candle_atual.name,
                        "entrada": entrada,
                        "stop": stop,
                        "alvo": alvo
                    })
                
                # Após encontrar um sinal, volta ao estado inicial para procurar a próxima oportunidade
                estado = "PROCURANDO_TENDENCIA"

    return pd.DataFrame(sinais)

def executar_simulacao(df_historico, df_sinais, tipo_operacao):
    """Simula os trades e retorna um DataFrame com os resultados."""
    resultados = []
    for _, sinal in df_sinais.iterrows():
        data_entrada = pd.to_datetime(sinal['data'])
        preco_entrada, stop, alvo = sinal['entrada'], sinal['stop'], sinal['alvo']
        df_futuro = df_historico[df_historico.index > data_entrada]
        resultado_trade = {"data_entrada": data_entrada, "preco_entrada": preco_entrada, "stop": stop, "alvo": alvo, "resultado": "Aberto", "data_saida": None, "preco_saida": None}
        for data_candle, candle in df_futuro.iterrows():
            if tipo_operacao == 'compra':
                if candle['low'] <= stop:
                    resultado_trade.update({"resultado": "Loss", "data_saida": data_candle, "preco_saida": stop}); break
                elif candle['high'] >= alvo:
                    resultado_trade.update({"resultado": "Gain", "data_saida": data_candle, "preco_saida": alvo}); break
            else: # Venda
                if candle['high'] >= stop:
                    resultado_trade.update({"resultado": "Loss", "data_saida": data_candle, "preco_saida": stop}); break
                elif candle['low'] <= alvo:
                    resultado_trade.update({"resultado": "Gain", "data_saida": data_candle, "preco_saida": alvo}); break
        resultados.append(resultado_trade)
    return pd.DataFrame(resultados)

def calcular_metricas(resultados_df, lotes, valor_por_ponto):
    """Calcula as métricas de performance do backtest."""
    if resultados_df.empty:
        return {}
    resultados_df['pnl_pontos'] = resultados_df.apply(lambda row: abs(row['preco_saida'] - row['preco_entrada']) if row['resultado'] == 'Gain' else -abs(row['preco_saida'] - row['preco_entrada']) if row['resultado'] == 'Loss' else 0, axis=1)
    resultados_df['pnl_financeiro'] = resultados_df['pnl_pontos'] * lotes * valor_por_ponto
    total_trades = len(resultados_df)
    trades_gain = resultados_df[resultados_df['resultado'] == 'Gain']
    trades_loss = resultados_df[resultados_df['resultado'] == 'Loss']
    taxa_acerto = (len(trades_gain) / total_trades) * 100 if total_trades > 0 else 0
    lucro_bruto = resultados_df['pnl_financeiro'].sum()
    media_gain = trades_gain['pnl_financeiro'].mean() if not trades_gain.empty else 0
    media_loss = abs(trades_loss['pnl_financeiro'].mean()) if not trades_loss.empty else 0
    risco_retorno = (media_gain / media_loss) if media_loss != 0 else float('inf')
    equity_curve = resultados_df['pnl_financeiro'].cumsum()
    pico_anterior = equity_curve.cummax()
    drawdown_financeiro = (pico_anterior - equity_curve).max()
    return {
        "totalOperacoes": total_trades,
        "tradesVencedores": len(trades_gain),
        "tradesPerdedores": len(trades_loss),
        "taxaAcerto": f"{taxa_acerto:.2f}%",
        "lucroBrutoTotal": f"$ {lucro_bruto:,.2f}",
        "riscoRetorno": f"{risco_retorno:.2f}" if risco_retorno != float('inf') else "N/A",
        "drawdownMaximo": f"$ {drawdown_financeiro:,.2f}"
    }
