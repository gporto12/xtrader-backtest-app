import pandas as pd
import numpy as np
import requests
from datetime import datetime

def buscar_dados_api(ativo, timeframe, data_inicio, data_fim, api_key):
    """Busca dados históricos da API da Polygon.io."""
    print(f"Buscando dados para {ativo} de {data_inicio} a {data_fim}...")
    
    # A API da Polygon espera o timeframe em minúsculas (day, hour, minute)
    # Vamos assumir 'day' para este exemplo
    timeframe_map = {'D1': 'day', 'H1': 'hour', 'M5': 'minute'}
    polygon_timeframe = timeframe_map.get(timeframe, 'day')
    
    # Para timeframe em minutos, o 'multiplier' seria 5
    multiplier = 5 if polygon_timeframe == 'minute' else 1

    API_URL = f"https://api.polygon.io/v2/aggs/ticker/{ativo}/range/{multiplier}/{polygon_timeframe}/{data_inicio}/{data_fim}"
    
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000, # Limite alto para capturar todos os dados no período
        'apiKey': api_key,
    }

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
    except Exception as e:
        print(f"Erro ao processar os dados: {e}")
        return None


def detectar_invert50_venda(df, mme_curta=9, mme_media=20, mme_longa=50, mme_geral=200, tolerancia_alinhamento=0.01, tolerancia_toque=0.01):
    """Detecta o padrão INVERT 50 (venda) com base em MMEs e candles de força."""
    df_copy = df.copy()
    # Usar adjust=False para um cálculo de MME mais consistente
    df_copy[f'MME{mme_curta}'] = df_copy['close'].ewm(span=mme_curta, adjust=False).mean()
    df_copy[f'MME{mme_media}'] = df_copy['close'].ewm(span=mme_media, adjust=False).mean()
    df_copy[f'MME{mme_longa}'] = df_copy['close'].ewm(span=mme_longa, adjust=False).mean()
    df_copy[f'MME{mme_geral}'] = df_copy['close'].ewm(span=mme_geral, adjust=False).mean()
    
    # Tolerâncias baseadas em percentual do preço para serem mais adaptáveis
    tolerancia_toque_valor = df_copy[f'MME{mme_longa}'] * tolerancia_toque
    
    df_copy['alinhadas'] = ((df_copy[f'MME{mme_longa}'] > df_copy[f'MME{mme_media}']) & (df_copy[f'MME{mme_media}'] > df_copy[f'MME{mme_curta}']))
    df_copy['tocou_longa'] = abs(df_copy['low'] - df_copy[f'MME{mme_longa}']) <= tolerancia_toque_valor
    df_copy['candle_forte'] = ((df_copy['close'].shift(-1) < df_copy[f'MME{mme_media}'].shift(-1)) & (df_copy['close'].shift(-1) < df_copy['open'].shift(-1)))
    df_copy['sinal_valido'] = df_copy['alinhadas'] & df_copy['tocou_longa'] & df_copy['candle_forte']
    
    sinais_df = df_copy[df_copy['sinal_valido']].copy()
    if sinais_df.empty: return pd.DataFrame()

    sinais_df['entrada'] = df_copy.loc[sinais_df.index, 'close'].shift(-1)
    sinais_df['stop'] = df_copy.loc[sinais_df.index, 'high'].shift(-1)
    
    mme_geral_entrada = df_copy.loc[sinais_df.index, f'MME{mme_geral}'].shift(-1)
    mme_geral_visivel = mme_geral_entrada > sinais_df['entrada']
    
    sinais_df['alvo'] = np.where(mme_geral_visivel, mme_geral_entrada, sinais_df['entrada'] - 2 * (sinais_df['stop'] - sinais_df['entrada']))
    
    resultado = sinais_df[['entrada', 'stop', 'alvo']].dropna().copy()
    resultado.reset_index(inplace=True)
    resultado.rename(columns={'datetime': 'data'}, inplace=True)
    
    return resultado[['data', 'entrada', 'stop', 'alvo']].round(2)

def executar_simulacao(df_historico, df_sinais):
    """Pega os sinais e simula o resultado de cada trade (gain ou loss)."""
    resultados = []
    if not isinstance(df_historico.index, pd.DatetimeIndex): df_historico.index = pd.to_datetime(df_historico.index)
    for _, sinal in df_sinais.iterrows():
        data_entrada = pd.to_datetime(sinal['data'])
        preco_entrada, stop, alvo = sinal['entrada'], sinal['stop'], sinal['alvo']
        df_futuro = df_historico[df_historico.index > data_entrada]
        resultado_trade = {"data_entrada": data_entrada, "preco_entrada": preco_entrada, "stop": stop, "alvo": alvo, "resultado": "Aberto", "data_saida": None, "preco_saida": None}
        for data_candle, candle in df_futuro.iterrows():
            if candle['high'] >= stop:
                resultado_trade.update({"resultado": "Loss", "data_saida": data_candle, "preco_saida": stop}); break
            elif candle['low'] <= alvo:
                resultado_trade.update({"resultado": "Gain", "data_saida": data_candle, "preco_saida": alvo}); break
        resultados.append(resultado_trade)
    return pd.DataFrame(resultados)

def rodar_backtest(df_historico, ativo, lotes, valor_por_ponto):
    """Orquestra todo o processo de backtest e retorna um DataFrame com os resultados."""
    print(f"\n--- Iniciando Backtest para o Ativo: {ativo} ---")
    sinais = detectar_invert50_venda(df_historico)
    if sinais.empty:
        print("Nenhum sinal encontrado para o período informado.")
        return None
        
    print(f"Total de Sinais Encontrados: {len(sinais)}")
    resultados_df = executar_simulacao(df_historico, sinais)
    
    resultados_df['pnl_pontos'] = resultados_df.apply(lambda row: abs(row['preco_saida'] - row['preco_entrada']) if row['resultado'] == 'Gain' else -abs(row['preco_saida'] - row['preco_entrada']) if row['resultado'] == 'Loss' else 0, axis=1)
    resultados_df['pnl_financeiro'] = resultados_df['pnl_pontos'] * lotes * valor_por_ponto
    
    resultados_df['equity_curve'] = resultados_df['pnl_financeiro'].cumsum()
    
    print("--- Backtest Concluído ---")
    return resultados_df.round(2)
