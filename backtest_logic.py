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
    """Detecta sinais de compra ou venda usando a lógica refinada 'Invert 50'."""
    if df is None or len(df) < 51:
        return pd.DataFrame()

    df_copy = df.copy()
    mme_curta, mme_media, mme_longa, mme_geral = 9, 20, 50, 200
    
    df_copy.ta.ema(length=mme_curta, append=True, col_names=(f'MME{mme_curta}',))
    df_copy.ta.ema(length=mme_media, append=True, col_names=(f'MME{mme_media}',))
    df_copy.ta.ema(length=mme_longa, append=True, col_names=(f'MME{mme_longa}',))
    df_copy.ta.ema(length=mme_geral, append=True, col_names=(f'MME{mme_geral}',))
    df_copy.dropna(inplace=True)

    sinais = []
    estado = "PROCURANDO_TENDENCIA"
    
    for i in range(len(df_copy)):
        candle = df_copy.iloc[i]
        
        tendencia_alta = candle[f'MME{mme_curta}'] > candle[f'MME{mme_media}'] > candle[f'MME{mme_longa}']
        tendencia_baixa = candle[f'MME{mme_longa}'] > candle[f'MME{mme_media}'] > candle[f'MME{mme_curta}']

        if estado == "PROCURANDO_TENDENCIA":
            if tipo_estrategia == 'compra' and tendencia_alta and candle['low'] <= candle[f'MME{mme_longa}']:
                estado = "PROCURANDO_GATILHO"
            elif tipo_estrategia == 'venda' and tendencia_baixa and candle['high'] >= candle[f'MME{mme_longa}']:
                estado = "PROCURANDO_GATILHO"
        
        elif estado == "PROCURANDO_GATILHO":
            gatilho_compra = tipo_estrategia == 'compra' and tendencia_alta and candle['close'] > candle['open'] and candle['close'] > candle[f'MME{mme_media}']
            gatilho_venda = tipo_estrategia == 'venda' and tendencia_baixa and candle['close'] < candle['open'] and candle['close'] < candle[f'MME{mme_media}']

            if gatilho_compra or gatilho_venda:
                entrada = candle['close']
                risco_multiplicador = 2.0
                
                if tipo_estrategia == 'compra':
                    stop = candle['low']
                    risco = entrada - stop
                    alvo = entrada + (risco * risco_multiplicador)
                else: # Venda
                    stop = candle['high']
                    risco = stop - entrada
                    alvo = entrada - (risco * risco_multiplicador)

                if risco > 0:
                    sinais.append({"data": candle.name, "entrada": entrada, "stop": stop, "alvo": alvo})
                
                estado = "PROCURANDO_TENDENCIA" # Reset para procurar o próximo
            
            # Reset se a tendência quebrar
            elif (tipo_estrategia == 'compra' and not tendencia_alta) or (tipo_estrategia == 'venda' and not tendencia_baixa):
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
            else:
                if candle['high'] >= stop:
                    resultado_trade.update({"resultado": "Loss", "data_saida": data_candle, "preco_saida": stop}); break
                elif candle['low'] <= alvo:
                    resultado_trade.update({"resultado": "Gain", "data_saida": data_candle, "preco_saida": alvo}); break
        resultados.append(resultado_trade)
    return pd.DataFrame(resultados)

def calcular_metricas(resultados_df, lotes, valor_por_ponto):
    """Calcula as métricas de performance do backtest."""
    if resultados_df.empty or 'resultado' not in resultados_df.columns:
        return {}

    resultados_df['pnl_financeiro'] = resultados_df.apply(
        lambda row: (abs(row['preco_saida'] - row['preco_entrada']) if row['resultado'] == 'Gain' else -abs(row['preco_saida'] - row['preco_entrada'])) * lotes * valor_por_ponto if row['resultado'] in ['Gain', 'Loss'] else 0, axis=1
    )
    
    trades_finalizados = resultados_df[resultados_df['resultado'] != 'Aberto']
    if trades_finalizados.empty: return {}

    total_trades = len(trades_finalizados)
    trades_gain = trades_finalizados[trades_finalizados['resultado'] == 'Gain']
    trades_loss = trades_finalizados[trades_finalizados['resultado'] == 'Loss']
    
    taxa_acerto = (len(trades_gain) / total_trades) * 100 if total_trades > 0 else 0
    lucro_bruto = trades_finalizados['pnl_financeiro'].sum()
    
    media_gain = trades_gain['pnl_financeiro'].mean() if not trades_gain.empty else 0
    media_loss = abs(trades_loss['pnl_financeiro'].mean()) if not trades_loss.empty else 0
    risco_retorno = (media_gain / media_loss) if media_loss != 0 else float('inf')

    equity_curve = trades_finalizados['pnl_financeiro'].cumsum()
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
