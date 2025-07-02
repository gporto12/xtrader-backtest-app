import pandas as pd
import numpy as np
import requests

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
        df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'}, inplace=True)
        df.set_index('datetime', inplace=True)
        print(f"Dados carregados com sucesso! Total de {len(df)} candles.")
        return df[['open', 'high', 'low', 'close']]
    except requests.exceptions.RequestException as e:
        print(f"Erro ao chamar a API da Polygon: {e}")
        return None

def detectar_sinais(df, tipo_estrategia):
    """Detecta sinais de compra ou venda com base na estratégia escolhida."""
    df_copy = df.copy()
    mme_curta, mme_media, mme_longa, mme_geral = 9, 20, 50, 200
    tolerancia_toque = 0.01

    df_copy[f'MME{mme_curta}'] = df_copy['close'].ewm(span=mme_curta, adjust=False).mean()
    df_copy[f'MME{mme_media}'] = df_copy['close'].ewm(span=mme_media, adjust=False).mean()
    df_copy[f'MME{mme_longa}'] = df_copy['close'].ewm(span=mme_longa, adjust=False).mean()
    df_copy[f'MME{mme_geral}'] = df_copy['close'].ewm(span=mme_geral, adjust=False).mean()
    
    tolerancia_toque_valor = df_copy[f'MME{mme_longa}'] * tolerancia_toque

    if tipo_estrategia == 'compra':
        df_copy['alinhadas'] = (df_copy[f'MME{mme_curta}'] > df_copy[f'MME{mme_media}']) & (df_copy[f'MME{mme_media}'] > df_copy[f'MME{mme_longa}'])
        df_copy['tocou_longa'] = abs(df_copy['low'] - df_copy[f'MME{mme_longa}']) <= tolerancia_toque_valor
        df_copy['candle_forte'] = (df_copy['close'].shift(-1) > df_copy[f'MME{mme_media}'].shift(-1)) & (df_copy['close'].shift(-1) > df_copy['open'].shift(-1))
    else: # Venda
        df_copy['alinhadas'] = (df_copy[f'MME{mme_longa}'] > df_copy[f'MME{mme_media}']) & (df_copy[f'MME{mme_media}'] > df_copy[f'MME{mme_curta}'])
        df_copy['tocou_longa'] = abs(df_copy['high'] - df_copy[f'MME{mme_longa}']) <= tolerancia_toque_valor
        df_copy['candle_forte'] = (df_copy['close'].shift(-1) < df_copy[f'MME{mme_media}'].shift(-1)) & (df_copy['close'].shift(-1) < df_copy['open'].shift(-1))

    df_copy['sinal_valido'] = df_copy['alinhadas'] & df_copy['tocou_longa'] & df_copy['candle_forte']
    sinais_df = df_copy[df_copy['sinal_valido']].copy()
    if sinais_df.empty: return pd.DataFrame()

    sinais_df['entrada'] = df_copy.loc[sinais_df.index, 'close'].shift(-1)
    mme_geral_entrada = df_copy.loc[sinais_df.index, f'MME{mme_geral}'].shift(-1)

    if tipo_estrategia == 'compra':
        sinais_df['stop'] = df_copy.loc[sinais_df.index, 'low'].shift(-1)
        sinais_df['alvo'] = np.where(mme_geral_entrada > sinais_df['entrada'], mme_geral_entrada, sinais_df['entrada'] + 2 * (sinais_df['entrada'] - sinais_df['stop']))
    else: # Venda
        sinais_df['stop'] = df_copy.loc[sinais_df.index, 'high'].shift(-1)
        sinais_df['alvo'] = np.where(mme_geral_entrada < sinais_df['entrada'], mme_geral_entrada, sinais_df['entrada'] - 2 * (sinais_df['stop'] - sinais_df['entrada']))

    resultado = sinais_df[['entrada', 'stop', 'alvo']].dropna().copy()
    resultado.reset_index(inplace=True)
    resultado.rename(columns={'datetime': 'data'}, inplace=True)
    return resultado[['data', 'entrada', 'stop', 'alvo']].round(2)

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
