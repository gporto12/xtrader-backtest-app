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
    """Detecta sinais de compra ou venda usando a lógica de REVERSÃO da estratégia Invert 50."""
    if df is None or len(df) < 200:
        return pd.DataFrame()

    df.ta.ema(length=9, append=True, col_names=('MME9',))
    df.ta.ema(length=20, append=True, col_names=('MME20',))
    df.ta.ema(length=50, append=True, col_names=('MME50',))
    df.ta.ema(length=200, append=True, col_names=('MME200',))
    df.dropna(inplace=True)

    sinais = []
    estado = "AGUARDANDO_REVERSAO" 
    
    df['MME20_acima_MME50'] = df['MME20'] > df['MME50']
    # CORREÇÃO: Garante que a série é booleana antes de usar o operador '~'
    shifted_series = df['MME20_acima_MME50'].shift(1).fillna(False).astype(bool)
    df['cruzou_para_cima'] = df['MME20_acima_MME50'] & ~shifted_series
    df['cruzou_para_baixo'] = ~df['MME20_acima_MME50'] & shifted_series

    for i in range(1, len(df)):
        candle = df.iloc[i]
        
        tendencia_alta_confirmada = candle['MME9'] > candle['MME20'] and candle['MME20'] > candle['MME50']
        tendencia_baixa_confirmada = candle['MME50'] > candle['MME20'] and candle['MME20'] > candle['MME9']

        if estado == "AGUARDANDO_REVERSAO":
            if tipo_estrategia == 'compra' and candle['cruzou_para_cima']:
                estado = "AGUARDANDO_PULLBACK"
            elif tipo_estrategia == 'venda' and candle['cruzou_para_baixo']:
                estado = "AGUARDANDO_PULLBACK"

        elif estado == "AGUARDANDO_PULLBACK":
            if tipo_estrategia == 'compra' and tendencia_alta_confirmada and candle['low'] <= candle['MME50']:
                estado = "AGUARDANDO_GATILHO"
            elif tipo_estrategia == 'venda' and tendencia_baixa_confirmada and candle['high'] >= candle['MME50']:
                estado = "AGUARDANDO_GATILHO"
        
        elif estado == "AGUARDANDO_GATILHO":
            gatilho_compra = tipo_estrategia == 'compra' and tendencia_alta_confirmada and candle['close'] > candle['open'] and candle['close'] > candle['MME20']
            gatilho_venda = tipo_estrategia == 'venda' and tendencia_baixa_confirmada and candle['close'] < candle['open'] and candle['close'] < candle['MME20']

            if gatilho_compra or gatilho_venda:
                entrada = candle['close']
                alvo = candle['MME200']
                
                if (tipo_estrategia == 'compra' and entrada < alvo) or (tipo_estrategia == 'venda' and entrada > alvo):
                    if tipo_estrategia == 'compra':
                        stop = candle['low']
                    else:
                        stop = candle['high']
                    
                    sinais.append({"data": candle.name, "entrada": entrada, "stop": stop, "alvo": alvo})
                    estado = "AGUARDANDO_REVERSAO"
                else:
                    estado = "AGUARDANDO_REVERSAO"
            
            elif (tipo_estrategia == 'compra' and not tendencia_alta_confirmada) or (tipo_estrategia == 'venda' and not tendencia_baixa_confirmada):
                estado = "AGUARDANDO_REVERSAO"

    return pd.DataFrame(sinais)

def executar_simulacao(df_historico, df_sinais, tipo_operacao):
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
    if resultados_df.empty or 'resultado' not in resultados_df.columns:
        return {}
    trades_finalizados = resultados_df[resultados_df['resultado'] != 'Aberto'].copy()
    if trades_finalizados.empty: return { "totalOperacoes": 0 }
    trades_finalizados['pnl_financeiro'] = trades_finalizados.apply(
        lambda row: (abs(row['preco_saida'] - row['preco_entrada']) if row['resultado'] == 'Gain' else -abs(row['preco_saida'] - row['preco_entrada'])) * lotes * valor_por_ponto, axis=1
    )
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
