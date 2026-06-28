import { View, Text, StyleSheet } from 'react-native';
import { Colors, Fonts } from '../constants/theme';

interface Props { name: string; size?: number }

export default function GuideAvatar({ name, size = 28 }: Props) {
  return (
    <View style={[styles.circle, { width: size, height: size, borderRadius: size / 2 }]}>
      <Text style={[styles.initial, { fontSize: size * 0.54 }]}>
        {name.charAt(0).toUpperCase()}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  circle: { backgroundColor: Colors.amber, alignItems: 'center', justifyContent: 'center' },
  initial: { fontFamily: Fonts.display, color: Colors.dark },
});
