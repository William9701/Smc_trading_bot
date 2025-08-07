# analysis_dashboard/structure_visualizer.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

from market_structure.swing_analyzer import SwingPoint
from market_structure.bos_choc_detector import StructureBreakPoint
from utils.constants import MarketStructure, StructureBreak

class StructureVisualizer:
    """
    Professional market structure visualization for debugging and analysis
    Creates interactive charts showing swings, BOS/CHoC, and trend analysis
    """
    
    def __init__(self):
        self.color_scheme = {
            'bullish': '#00ff88',
            'bearish': '#ff4444', 
            'neutral': '#888888',
            'swing_high': '#ff6b6b',
            'swing_low': '#4ecdc4',
            'bos': '#ffd93d',
            'choc': '#ff6348',
            'background': '#1e1e1e',
            'grid': '#333333'
        }
        
        logger.info("Structure Visualizer initialized")
    
    def create_structure_chart(
        self, 
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        structure_breaks: List[StructureBreakPoint],
        current_structure: Dict,
        symbol: str = "Unknown",
        timeframe: str = "Unknown"
    ) -> go.Figure:
        """
        Create comprehensive structure analysis chart
        """
        try:
            # Create subplot with secondary y-axis for indicators
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{symbol} {timeframe} - Market Structure Analysis", "Volume"),
                vertical_spacing=0.05,
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price",
                    increasing_line_color=self.color_scheme['bullish'],
                    decreasing_line_color=self.color_scheme['bearish'],
                    increasing_fillcolor=self.color_scheme['bullish'],
                    decreasing_fillcolor=self.color_scheme['bearish']
                ),
                row=1, col=1
            )
            
            # Add swing points
            self._add_swing_points(fig, swing_points, row=1)
            
            # Add structure breaks
            self._add_structure_breaks(fig, structure_breaks, df, row=1)
            
            # Add trend lines
            self._add_trend_lines(fig, swing_points, current_structure, row=1)
            
            # Add volume
            if 'volume' in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['volume'],
                        name="Volume",
                        marker_color=self.color_scheme['neutral'],
                        opacity=0.6
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f"SMC Structure Analysis: {symbol} ({timeframe})",
                template="plotly_dark",
                plot_bgcolor=self.color_scheme['background'],
                paper_bgcolor=self.color_scheme['background'],
                font=dict(color='white'),
                height=800,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left", 
                    x=0.01
                )
            )
            
            # Update axes
            fig.update_xaxes(
                gridcolor=self.color_scheme['grid'],
                showgrid=True,
                rangeslider_visible=False
            )
            fig.update_yaxes(
                gridcolor=self.color_scheme['grid'],
                showgrid=True
            )
            
            # Add structure info annotation
            structure_text = self._create_structure_info_text(current_structure)
            fig.add_annotation(
                text=structure_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="white",
                borderwidth=1,
                font=dict(size=12, color="white"),
                align="left"
            )
            
            logger.info(f"Structure chart created for {symbol} {timeframe}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating structure chart: {e}")
            return self._create_error_chart(str(e))
    
    def _add_swing_points(self, fig: go.Figure, swing_points: List[SwingPoint], row: int):
        """Add swing points to the chart"""
        if not swing_points:
            return
        
        # Separate highs and lows
        swing_highs = [sp for sp in swing_points if sp.swing_type == 'HIGH']
        swing_lows = [sp for sp in swing_points if sp.swing_type == 'LOW']
        
        # Add swing highs
        if swing_highs:
            fig.add_trace(
                go.Scatter(
                    x=[sp.timestamp for sp in swing_highs],
                    y=[sp.price for sp in swing_highs],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color=self.color_scheme['swing_high'],
                        line=dict(width=2, color='white')
                    ),
                    name='Swing Highs',
                    text=[f"H: {sp.price:.5f}<br>Strength: {sp.strength:.2f}" for sp in swing_highs],
                    hovertemplate="<b>Swing High</b><br>%{text}<br>Time: %{x}<extra></extra>"
                ),
                row=row, col=1
            )
        
        # Add swing lows
        if swing_lows:
            fig.add_trace(
                go.Scatter(
                    x=[sp.timestamp for sp in swing_lows],
                    y=[sp.price for sp in swing_lows],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color=self.color_scheme['swing_low'],
                        line=dict(width=2, color='white')
                    ),
                    name='Swing Lows',
                    text=[f"L: {sp.price:.5f}<br>Strength: {sp.strength:.2f}" for sp in swing_lows],
                    hovertemplate="<b>Swing Low</b><br>%{text}<br>Time: %{x}<extra></extra>"
                ),
                row=row, col=1
            )
    
    def _add_structure_breaks(self, fig: go.Figure, structure_breaks: List[StructureBreakPoint], df: pd.DataFrame, row: int):
        """Add structure breaks to the chart"""
        if not structure_breaks:
            return
        
        for break_point in structure_breaks:
            # Determine color and symbol based on break type
            if 'BOS' in break_point.break_type:
                color = self.color_scheme['bos']
                symbol = 'star'
                name_prefix = 'BOS'
            else:  # CHoC
                color = self.color_scheme['choc']
                symbol = 'diamond'
                name_prefix = 'CHoC'
            
            # Add break point marker
            fig.add_trace(
                go.Scatter(
                    x=[break_point.timestamp],
                    y=[break_point.price],
                    mode='markers',
                    marker=dict(
                        symbol=symbol,
                        size=15,
                        color=color,
                        line=dict(width=2, color='white')
                    ),
                    name=f"{name_prefix} ({'Bullish' if 'BULLISH' in break_point.break_type else 'Bearish'})",
                    text=f"{break_point.break_type}<br>Price: {break_point.price:.5f}<br>Strength: {break_point.confirmation_strength:.2f}",
                    hovertemplate="<b>%{text}</b><br>Time: %{x}<extra></extra>",
                    showlegend=True
                ),
                row=row, col=1
            )
            
            # Add vertical line to highlight break
            fig.add_vline(
                x=break_point.timestamp,
                line=dict(color=color, width=1, dash="dash"),
                opacity=0.5,
                row=row, col=1
            )
    
    def _add_trend_lines(self, fig: go.Figure, swing_points: List[SwingPoint], current_structure: Dict, row: int):
        """Add trend lines based on swing points"""
        if len(swing_points) < 4:
            return
        
        try:
            # Get recent swing points for trend line
            recent_highs = [sp for sp in swing_points[-10:] if sp.swing_type == 'HIGH']
            recent_lows = [sp for sp in swing_points[-10:] if sp.swing_type == 'LOW']
            
            current_trend = current_structure.get('trend', MarketStructure.UNKNOWN)
            
            # Draw trend lines based on current structure
            if current_trend == MarketStructure.BULLISH and len(recent_lows) >= 2:
                # Draw bullish trend line connecting recent lows
                self._draw_trend_line(
                    fig, recent_lows[-2:], self.color_scheme['bullish'], "Bullish Trend", row
                )
            
            elif current_trend == MarketStructure.BEARISH and len(recent_highs) >= 2:
                # Draw bearish trend line connecting recent highs
                self._draw_trend_line(
                    fig, recent_highs[-2:], self.color_scheme['bearish'], "Bearish Trend", row
                )
                
        except Exception as e:
            logger.debug(f"Error adding trend lines: {e}")
    
    def _draw_trend_line(self, fig: go.Figure, points: List[SwingPoint], color: str, name: str, row: int):
        """Draw a trend line connecting swing points"""
        if len(points) < 2:
            return
        
        # Sort points by timestamp
        sorted_points = sorted(points, key=lambda x: x.timestamp)
        
        # Calculate trend line extension
        x1, y1 = sorted_points[0].timestamp, sorted_points[0].price
        x2, y2 = sorted_points[-1].timestamp, sorted_points[-1].price
        
        # Extend line forward
        time_diff = x2 - x1
        price_diff = y2 - y1
        
        # Extend 20% forward
        x3 = x2 + time_diff * 0.2
        if time_diff.total_seconds() != 0:
            y3 = y2 + price_diff * 0.2
        else:
            y3 = y2
        
        fig.add_trace(
            go.Scatter(
                x=[x1, x2, x3],
                y=[y1, y2, y3],
                mode='lines',
                line=dict(color=color, width=2, dash='solid'),
                name=name,
                opacity=0.8
            ),
            row=row, col=1
        )
    
    def _create_structure_info_text(self, current_structure: Dict) -> str:
        """Create structure information text"""
        trend = current_structure.get('trend', 'Unknown')
        confidence = current_structure.get('confidence', 0.0)
        last_break = current_structure.get('break_type', 'None')
        breaks_in_direction = current_structure.get('breaks_in_direction', 0)
        
        return f"""<b>Current Structure</b>
Trend: {trend}
Confidence: {confidence:.1%}
Last Break: {last_break}
Consecutive Breaks: {breaks_in_direction}"""
    
    def _create_error_chart(self, error_msg: str) -> go.Figure:
        """Create error chart when visualization fails"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"Chart Error:<br>{error_msg}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            font=dict(size=16, color="red"),
            showarrow=False
        )
        
        fig.update_layout(
            title="Chart Generation Error",
            template="plotly_dark"
        )
        
        return fig
    
    def create_multi_timeframe_chart(
        self, 
        multi_tf_data: Dict,
        symbol: str = "Unknown"
    ) -> go.Figure:
        """Create multi-timeframe structure comparison chart"""
        try:
            timeframes = list(multi_tf_data.keys())
            num_timeframes = len(timeframes)
            
            if num_timeframes == 0:
                return self._create_error_chart("No timeframe data provided")
            
            # Create subplots for each timeframe
            fig = make_subplots(
                rows=num_timeframes, cols=1,
                subplot_titles=[f"{symbol} - {tf}" for tf in timeframes],
                vertical_spacing=0.05,
                shared_xaxes=True
            )
            
            for i, (tf, tf_data) in enumerate(multi_tf_data.items(), 1):
                df = tf_data['price_data']
                swing_points = tf_data.get('swing_points', [])
                structure_breaks = tf_data.get('structure_breaks', [])
                current_structure = tf_data.get('current_structure', {})
                
                # Add candlestick for this timeframe
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=f"{tf} Price",
                        increasing_line_color=self.color_scheme['bullish'],
                        decreasing_line_color=self.color_scheme['bearish']
                    ),
                    row=i, col=1
                )
                
                # Add swing points for this timeframe
                self._add_swing_points(fig, swing_points, row=i)
                
                # Add structure breaks for this timeframe  
                self._add_structure_breaks(fig, structure_breaks, df, row=i)
                
                # Add structure info
                structure_text = self._create_structure_info_text(current_structure)
                fig.add_annotation(
                    text=structure_text,
                    xref="paper", yref=f"y{i} domain",
                    x=0.02, y=0.98,
                    bgcolor="rgba(0,0,0,0.8)",
                    bordercolor="white",
                    borderwidth=1,
                    font=dict(size=10, color="white"),
                    align="left"
                )
            
            # Update layout
            fig.update_layout(
                title=f"Multi-Timeframe Structure Analysis: {symbol}",
                template="plotly_dark",
                height=300 * num_timeframes,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating multi-timeframe chart: {e}")
            return self._create_error_chart(str(e))
    
    def create_structure_summary_dashboard(
        self,
        analysis_results: Dict,
        symbol: str = "Unknown"
    ) -> go.Figure:
        """Create structure analysis summary dashboard"""
        try:
            # Create 2x2 subplot grid
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Swing Detection Quality",
                    "Structure Break Types", 
                    "Trend Confidence Over Time",
                    "Multi-Timeframe Alignment"
                ],
                specs=[
                    [{"type": "indicator"}, {"type": "pie"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ]
            )
            
            # 1. Swing Detection Quality Gauge
            swing_quality = analysis_results.get('avg_swing_quality', 0.5)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=swing_quality * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Swing Quality %"},
                    delta={'reference': 80},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=1, col=1
            )
            
            # 2. Structure Break Types Pie Chart
            break_types = analysis_results.get('break_type_distribution', {
                'BOS Bullish': 15,
                'BOS Bearish': 12,
                'CHoC Bullish': 8,
                'CHoC Bearish': 10
            })
            
            fig.add_trace(
                go.Pie(
                    labels=list(break_types.keys()),
                    values=list(break_types.values()),
                    hole=0.3,
                    marker_colors=[
                        self.color_scheme['bullish'],
                        self.color_scheme['bearish'],
                        self.color_scheme['bos'],
                        self.color_scheme['choc']
                    ]
                ),
                row=1, col=2
            )
            
            # 3. Trend Confidence Over Time
            confidence_data = analysis_results.get('confidence_timeline', [])
            if confidence_data:
                fig.add_trace(
                    go.Scatter(
                        x=[item['timestamp'] for item in confidence_data],
                        y=[item['confidence'] for item in confidence_data],
                        mode='lines+markers',
                        name='Trend Confidence',
                        line=dict(color=self.color_scheme['bullish'], width=2)
                    ),
                    row=2, col=1
                )
            
            # 4. Multi-Timeframe Alignment
            tf_alignment = analysis_results.get('timeframe_alignment', {
                'M15': 0.85,
                'H1': 0.92, 
                'H4': 0.78
            })
            
            fig.add_trace(
                go.Bar(
                    x=list(tf_alignment.keys()),
                    y=list(tf_alignment.values()),
                    marker_color=self.color_scheme['neutral']
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f"Structure Analysis Dashboard: {symbol}",
                template="plotly_dark",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating summary dashboard: {e}")
            return self._create_error_chart(str(e))
    
    def save_chart_html(self, fig: go.Figure, filename: str, directory: str = "charts"):
        """Save chart as HTML file"""
        try:
            from pathlib import Path
            
            chart_dir = Path(directory)
            chart_dir.mkdir(exist_ok=True)
            
            filepath = chart_dir / f"{filename}.html"
            fig.write_html(str(filepath))
            
            logger.info(f"Chart saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            return None
    
    def export_chart_image(self, fig: go.Figure, filename: str, format: str = "png", directory: str = "charts"):
        """Export chart as image file"""
        try:
            from pathlib import Path
            
            chart_dir = Path(directory)
            chart_dir.mkdir(exist_ok=True)
            
            filepath = chart_dir / f"{filename}.{format}"
            fig.write_image(str(filepath))
            
            logger.info(f"Chart image saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting chart image: {e}")
            return None

# Factory function
def create_structure_visualizer() -> StructureVisualizer:
    """Factory function to create structure visualizer"""
    return StructureVisualizer()

# Usage example function
def demo_structure_visualization():
    """Demo function showing how to use the structure visualizer"""
    try:
        from data_service.data_fetcher import EnhancedMT5DataFetcher
        from market_structure.swing_analyzer import SwingAnalyzer
        from market_structure.bos_choc_detector import BOSCHOCDetector
        
        # Initialize components
        data_fetcher = EnhancedMT5DataFetcher()
        swing_analyzer = SwingAnalyzer()
        structure_detector = BOSCHOCDetector()
        visualizer = StructureVisualizer()
        
        # Initialize MT5 and fetch data
        if not data_fetcher.initialize_mt5():
            print("Failed to initialize MT5")
            return
        
        # Get data
        df = data_fetcher.get_symbol_data("EURUSD", "H1", 500)
        if df.empty:
            print("No data received")
            return
        
        # Analyze structure
        swing_analysis = swing_analyzer.analyze_swings(df, "EURUSD")
        structure_analysis = structure_detector.detect_structure_breaks(df, "EURUSD")
        
        if swing_analysis['success'] and structure_analysis['success']:
            # Create chart
            fig = visualizer.create_structure_chart(
                df=df,
                swing_points=swing_analysis['swing_points'],
                structure_breaks=structure_analysis['structure_breaks'],
                current_structure=structure_analysis['current_structure'],
                symbol="EURUSD",
                timeframe="H1"
            )
            
            # Save chart
            chart_path = visualizer.save_chart_html(fig, "eurusd_h1_structure_demo")
            print(f"Demo chart saved to: {chart_path}")
            
            # Show chart in browser (if available)
            try:
                fig.show()
            except:
                print("Could not display chart in browser")
        
        else:
            print("Analysis failed")
        
        # Cleanup
        data_fetcher.shutdown_mt5()
        
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    demo_structure_visualization()